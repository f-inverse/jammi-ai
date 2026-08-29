#!/usr/bin/env python3
"""check_lead_gate.py — the lead-proactivity gate's CI leg (v3, core).

`--self-test` drives the REAL hook scripts under
`.claude/hooks/lead-gate-{start,stop,pre}.sh` against fixture payloads in a
TEMP `.jammi/gate-state/` (via `CLAUDE_PROJECT_DIR`). v3 narrows the gate to
ONE choke point after round 1 and round 2 both found every free-text
predicate (site regexes, worktree/sha/token scans, write-verb walks, tag
scans) was jammable or dodgeable in its own way — a mechanism change, not a
third patch (`CONTRACT-v3.md`), further narrowed to a CORE under a
usage-limit scope cut: SendMessage gating, implementer-dispatch binding, and
the Bash backstop are DROPPED ENTIRELY (not log-only) — see the module doc
of `lead-gate-lib.py` for the exact boundary.

Required fixtures (RED when the corresponding hook arm is removed):

  G1  first dispatch of any verifier type is never gated
  G2  second round denied: prompt names the recorded worktree
  G3  second round denied: prompt names the recorded head_sha (full)
  G4  second round denied: prompt names the head_sha's first 7 chars
  G5  second round denied: prompt names the exact unit_branch
  G6  second round ALLOWED with an accepted relay artifact (full coverage)
  G7  a relay artifact missing one required site is NOT accepted -> deny
  G8  enumeration_missing: a relay artifact with a disjoint >=2-entry probe
      IS accepted; a probe entry equal to a finding location is NOT
  G9  a relay artifact with a mismatched block_ts/agent_type/unit_branch is
      NOT accepted
  G10 DODGE-5: an unlabeled re-dispatch (names none of the 3 anchors) is
      allowed — the documented residual
  G11 cross-type non-interference: a pressure-tester REFINE does not gate
      the first adversarial-audit dispatch (built non-vacuously)
  G12 a SAME-agent_type PASS clears its own BLOCK
  G13 a fix-verifier PASS clears an OLDER adversarial-audit BLOCK only when
      that BLOCK's relay artifact was accepted (not when it wasn't)
  G14 an UNPARSEABLE latest row gates exactly like a BLOCK (`is_open`
      covers both values — audit-r3 finding 5's surviving mutant)
  G15 anchors bind as WHOLE TOKENS, never raw substrings (audit-r3 finding
      1): an open BLOCK on `ci/gpu` does NOT gate `ci/gpu-dev`, `<worktree>2`
      and a 7-char-lookalike hex token do NOT gate; a path UNDER the
      recorded worktree and a TRUE >=7-char sha prefix still DO
  L1  closed-world agent-type lattice: unrecognized type -> deny
  L2  every `.claude/agents/*.md` card (+ harness built-ins) is classified;
      NEVER_GATED members carry no Edit/Write/MultiEdit in `tools:`
  L3  the agent-type field is read under every known spelling
      (`agent_type` works like `subagent_type`); a dispatch payload with NO
      agent-type field at all is a DISTINCT deny arm with its own remedy,
      never the unknown-type arm (audit-r3 finding 4)
  V1  a schema-template block quoted after a real one -> UNPARSEABLE(template)
  V2  a truncated fenced ```json block (no closing brace) -> UNPARSEABLE
      row IS written, never silently dropped
  V3  an unrecognized raw verdict value defaults to BLOCK and the deny
      reason names the raw value
  V4/V5/V6  `_PASS_LIKE` pinned individually: "PASS", "verified", "PROCEED"
  V7  a `</verdict>` (or a stray `}`) inside the verdict's own `notes`
      STRING does not truncate/corrupt the region (round-2 finding 6)
  V8  an UNPARSEABLE verdict is filed under the agent's SubagentStart
      binding, never a shared UNBOUND bucket, when a binding exists
  V9  a fenced ```json block with `kind` != "verdict" is UNPARSEABLE (does
      NOT fall back to the tag walk when a fenced block IS present)
  V10 the OPENING-marker scan is string-aware (audit-r3 finding 2): a PASS
      whose `notes` STRING mentions the ```json marker is recorded as PASS,
      not UNPARSEABLE
  E1  non-UTF-8 payload -> exit 2 (never 1)
  E2  python3 absent from PATH -> exit 2 (never 1)
  E3  a python3 that exits non-zero for an unrelated reason -> the wrapper
      still maps it to exit 2, never propagates the raw code
  E4  a UTF-8-valid but JSON-invalid payload (and a JSON non-object) ->
      `pre` exits 2, never coerced to `{}` and allowed (audit-r3 finding
      3); `stop` stays a best-effort writer and exits 0
  N6  no open BLOCK anywhere -> nothing gated
  N7  wall time < 1s per invocation
  R10 wiring: SubagentStart/SubagentStop/PreToolUse(Agent|Task) present,
      scripts executable, permissions.deny covers the hook files

Run: `python3 ci/scripts/check_lead_gate.py --self-test`
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / ".claude" / "hooks"
AGENTS_DIR = REPO_ROOT / ".claude" / "agents"
SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"
LEAD_GATE_LIB = HOOKS_DIR / "lead-gate-lib.py"

_WALL_TIMES: list[float] = []


class Failure(Exception):
    pass


def _run(script: str, payload: dict | bytes, project_dir: Path,
         env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    script_path = HOOKS_DIR / script
    data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode("utf-8")
    start = time.monotonic()
    proc = subprocess.run(
        ["/bin/sh", str(script_path)],
        input=data,
        capture_output=True,
        env=env,
        timeout=10,
    )
    _WALL_TIMES.append(time.monotonic() - start)
    proc.stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else proc.stdout
    proc.stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
    return proc


def _fresh_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="lead-gate-selftest-"))


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}: {detail}")


def _read_only_row(root: Path, unit_slug: str) -> dict:
    f = root / ".jammi" / "gate-state" / f"{unit_slug}.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows[-1]


def _write_block_row(root: Path, unit_branch: str, agent_id: str, agent_type: str,
                      class_enumeration: list[str] | None, findings_locations: list[str],
                      extra: dict | None = None, verdict: str = "BLOCK") -> dict:
    """Writes a real verdict row through the real lead-gate-stop.sh and
    returns the parsed row (so callers can read its exact `ts` for
    constructing a relay artifact)."""
    v: dict = {
        "kind": "verdict", "agent": agent_type, "diff_range": "base...head",
        "verdict": verdict, "uncertain": False, "unit_branch": unit_branch,
        "head_sha": "cafef00d1234567890abcdef1234567890abcdef",
        "worktree": f"/Users/x/worktrees/agent-{agent_type}",
        "findings": [
            {"axis": "x", "location": loc, "claim": "c", "stands": True, "severity": "block"}
            for loc in findings_locations
        ],
        "notes": "n",
    }
    if class_enumeration is not None:
        v["class_enumeration"] = class_enumeration
        v["sweep_method"] = "grep -n the pattern"
        v["exhaustive"] = True
    if extra:
        v.update(extra)
    msg = "Findings below.\n```json\n" + json.dumps(v) + "\n```\n"
    p = _run("lead-gate-stop.sh", {
        "agent_id": agent_id, "agent_type": agent_type, "last_assistant_message": msg,
    }, root)
    _assert(p.returncode == 0, "setup: lead-gate-stop.sh must exit 0", f"got {p.returncode}: {p.stderr}")
    return _read_only_row(root, _slug(unit_branch))


def _slug(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", branch.strip()) or "UNBOUND"


def _relay_path_exact(root: Path, row: dict) -> Path:
    """The EXACT path lead-gate-lib.py's `relay_artifact_path()` computes,
    via the real module (never a reimplementation), so fixtures write to
    the filename the hook will actually look for."""
    import importlib.util
    if "lead_gate_lib_v3" in sys.modules:
        mod = sys.modules["lead_gate_lib_v3"]
    else:
        spec = importlib.util.spec_from_file_location("lead_gate_lib_v3", LEAD_GATE_LIB)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sys.modules["lead_gate_lib_v3"] = mod
    return mod.relay_artifact_path(root / ".jammi" / "gate-state", _slug(row["unit_branch"]),
                                    row["agent_type"], row["ts"])


def _write_relay_exact(root: Path, row: dict, sites: dict[str, str] | None = None,
                        probe: list[str] | None = None, override: dict | None = None) -> None:
    path = _relay_path_exact(root, row)
    artifact = {"unit_branch": row["unit_branch"], "agent_type": row["agent_type"], "block_ts": row["ts"]}
    if sites is not None:
        artifact["sites"] = sites
    if probe is not None:
        artifact["probe"] = probe
    if override:
        artifact.update(override)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))


# ==========================================================================
# G1-G13 — the one gate
# ==========================================================================

def fixture_g1_first_round_never_gated() -> None:
    """No PRIOR adversarial-audit row exists anywhere -> the dispatch is
    structurally the first round of this type, allowed regardless of what
    the prompt names."""
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": "unit: feat/g1\nFIRST audit of the implemented diff, worktree /w/agent-g1",
    }}, root)
    _assert(p.returncode == 0, "G1", f"expected allow, got {p.returncode}: {p.stderr}")


def fixture_g2_second_round_denied_worktree() -> None:
    """Isolated to the worktree anchor ONLY — the prompt names neither the
    unit_branch nor the sha, so this cannot pass via a different anchor."""
    root = _fresh_root()
    row = _write_block_row(root, "feat/g2", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": f"re-audit at {row['worktree']}",
    }}, root)
    _assert(p.returncode == 2, "G2", f"expected deny(2), got {p.returncode}")


def fixture_g3_second_round_denied_full_sha() -> None:
    """Isolated to the full-sha anchor ONLY (no worktree/unit_branch in
    the prompt)."""
    root = _fresh_root()
    row = _write_block_row(root, "feat/g3", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": f"re-audit commit {row['head_sha']} please",
    }}, root)
    _assert(p.returncode == 2, "G3", f"expected deny(2), got {p.returncode}")


def fixture_g4_second_round_denied_short_sha() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g4", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    short = row["head_sha"][:7]
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": f"re-audit at {short} please",
    }}, root)
    _assert(p.returncode == 2, "G4", f"expected deny(2), got {p.returncode}")


def fixture_g5_second_round_denied_unit_branch() -> None:
    root = _fresh_root()
    _write_block_row(root, "feat/g5", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g5 now"}}, root)
    _assert(p.returncode == 2, "G5", f"expected deny(2), got {p.returncode}")


def fixture_g6_second_round_allowed_with_accepted_relay() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g6", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: feat/g6"}}, root)
    _assert(p.returncode == 0, "G6", f"expected allow, got {p.returncode}: {p.stderr}")


def fixture_g7_relay_missing_site_not_accepted() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g7", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"})  # missing b.py:2
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g7"}}, root)
    _assert(p.returncode == 2, "G7", f"expected deny(2), got {p.returncode}")


def fixture_g8_enumeration_missing_probe_fallback() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g8a", "a1", "adversarial-audit", None, ["foo.py:10"])
    _write_relay_exact(root, row, probe=["foo.py:10", "bar.py:5"])  # dup finding location
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g8a"}}, root)
    _assert(p.returncode == 2, "G8a", f"dup finding location must not accept, got {p.returncode}")

    root2 = _fresh_root()
    row2 = _write_block_row(root2, "feat/g8b", "a1", "adversarial-audit", None, ["foo.py:10"])
    _write_relay_exact(root2, row2, probe=["bar.py:5", "baz.py:9"])  # disjoint, >=2
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g8b"}}, root2)
    _assert(p.returncode == 0, "G8b", f"disjoint >=2-entry probe must accept, got {p.returncode}: {p.stderr}")


def fixture_g9_mismatched_relay_not_accepted() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g9", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, override={"block_ts": "wrong-ts"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g9"}}, root)
    _assert(p.returncode == 2, "G9", f"a mismatched block_ts must not accept, got {p.returncode}")


def fixture_g10_dodge5_unlabeled_redispatch_allowed() -> None:
    root = _fresh_root()
    _write_block_row(root, "feat/g10", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "unit: none\nre-audit the fix"}}, root)
    _assert(p.returncode == 0, "G10", f"documented residual must allow, got {p.returncode}: {p.stderr}")


def fixture_g11_cross_type_non_interference() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "ci/lead-proactivity-gate", "pt1", "pressure-tester",
                            ["CONTRACT.md:12"], ["CONTRACT.md:12"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": f"unit: ci/lead-proactivity-gate\nFIRST audit at {row['worktree']}",
    }}, root)
    _assert(p.returncode == 0, "G11", f"a pressure-tester REFINE must not gate the first adversarial-audit, got {p.returncode}: {p.stderr}")


def fixture_g12_same_type_pass_clears() -> None:
    root = _fresh_root()
    row = _write_block_row(root, "feat/g12", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {row['worktree']}"}}, root)
    _assert(p.returncode == 2, "G12 pre-PASS", "expected deny before PASS")
    _write_block_row(root, "feat/g12", "a2", "adversarial-audit", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {row['worktree']}"}}, root)
    _assert(p.returncode == 0, "G12", f"a same-type PASS must clear, got {p.returncode}: {p.stderr}")


def fixture_g13_verifier_pass_clears_audited_block_only_with_relay() -> None:
    root = _fresh_root()
    aa_row = _write_block_row(root, "feat/g13", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    # No relay yet: a fix-verifier PASS must NOT clear the audit BLOCK.
    _write_block_row(root, "feat/g13", "f1", "fix-verifier", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {aa_row['worktree']}"}}, root)
    _assert(p.returncode == 2, "G13a", f"fix-verifier PASS with NO relay must not clear, got {p.returncode}")

    root2 = _fresh_root()
    aa_row2 = _write_block_row(root2, "feat/g13b", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_relay_exact(root2, aa_row2, sites={"a.py:1": "fixed"})
    _write_block_row(root2, "feat/g13b", "f1", "fix-verifier", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {aa_row2['worktree']}"}}, root2)
    _assert(p.returncode == 0, "G13b", f"fix-verifier PASS WITH an accepted relay must clear, got {p.returncode}: {p.stderr}")
    # NOTE: the cross-type-clearing arm's chronological-order check
    # (`vidx > aa_idx` in `_adversarial_audit_cleared_by_verifier_pass`) has
    # NO independently observable effect via this gate: whenever a relay
    # artifact is accepted for the audit's OWN block, the second round is
    # already unlocked directly (the ONE GATE's base rule), so "cleared via
    # a stale cross-type PASS" and "unlocked via direct relay acceptance"
    # converge to the identical ALLOW outcome in every reachable state.
    # Implemented per the letter of the requirement; documented honestly
    # as unmutation-tested for this reason in the commit message, not
    # silently claimed covered.


def fixture_g14_unparseable_row_gates_like_block() -> None:
    """audit-r3 finding 5: mutating `is_open` so UNPARSEABLE no longer
    gates left the whole self-test green. This arm dies with that mutant:
    a unit whose ONLY row is UNPARSEABLE (no BLOCK row anywhere) must gate
    a same-type second dispatch that names its unit_branch."""
    root = _fresh_root()
    p = _run("lead-gate-start.sh", {"agent_id": "g14", "agent_type": "adversarial-audit",
                                     "prompt": "unit: feat/g14\naudit please"}, root)
    _assert(p.returncode == 0, "G14 setup", "start must exit 0")
    p = _run("lead-gate-stop.sh", {"agent_id": "g14", "agent_type": "adversarial-audit",
                                    "last_assistant_message": "ran out of context, no verdict"}, root)
    _assert(p.returncode == 0, "G14 setup", "stop must exit 0")
    row = _read_only_row(root, "feat_g14")
    _assert(row["verdict"] == "UNPARSEABLE", "G14 setup", f"expected UNPARSEABLE, got {row['verdict']!r}")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g14 now"}}, root)
    _assert(p.returncode == 2, "G14",
            f"an UNPARSEABLE latest row must gate exactly like a BLOCK, got {p.returncode}: {p.stderr}")


def fixture_g15_whole_token_anchors_never_raw_substrings() -> None:
    """audit-r3 finding 1: `ub in text` was a raw substring match, so an
    open BLOCK on `ci/gpu` denied the FIRST audit of `ci/gpu-dev`. Anchors
    bind as whole tokens: near-miss longer tokens ALLOW; a path UNDER the
    recorded worktree and a TRUE >=7-char sha prefix still DENY."""
    root = _fresh_root()
    row = _write_block_row(root, "ci/gpu", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])

    # Near misses — every one of these was a false DENY under substring matching.
    for label, prompt in (
        ("unit-prefix", "unit: ci/gpu-dev\nFIRST audit of the gpu-dev unit"),
        ("worktree-prefix", f"FIRST audit at {row['worktree']}2"),
        ("sha-lookalike", f"FIRST audit of commit {row['head_sha'][:7]}9 (a different commit)"),
    ):
        p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": "adversarial-audit", "prompt": prompt}}, root)
        _assert(p.returncode == 0, "G15",
                f"[{label}] a longer token merely CONTAINING the anchor must allow, got {p.returncode}: {p.stderr}")

    # True references — still denied.
    for label, prompt in (
        ("path-under-worktree", f"re-audit the diff at {row['worktree']}/crates/x"),
        ("8-char-sha-prefix", f"re-audit commit {row['head_sha'][:8]} please"),
        ("exact-unit", "re-audit unit: ci/gpu now"),
    ):
        p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": "adversarial-audit", "prompt": prompt}}, root)
        _assert(p.returncode == 2, "G15",
                f"[{label}] a true whole-token reference must still deny, got {p.returncode}: {p.stderr}")


# ==========================================================================
# T1-T4 — "a compliant verdict binds to its unit" (capability commit 1).
# ==========================================================================

_CARD_NAMES = (
    "adversarial-audit", "acceptance-verifier", "citation-checker",
    "discipline-test-auditor", "fix-verifier", "oracle", "pressure-tester",
)
_VERIFIER_SECOND_ROUND = {"adversarial-audit", "fix-verifier", "acceptance-verifier"}

# The literal shape every card's own schema line now instructs (a BARE
# `unit_branch` token plus a separate `unit_branch_source` provenance
# note) — matched against the REAL card text, never hand-typed, so this
# fixture tracks whatever the card's CURRENT wording is rather than a
# hand-typed guess (the failure mode that let the original bug — the
# annotated "say which" shape being rejected as a template — ship
# unnoticed: the pre-existing V1 fixture used a synthetic `"<branch>"`
# stand-in that never matched what a real verifier actually produces).
_CARD_UNIT_BRANCH_LINE_RE = re.compile(
    r'^[ \t]*"unit_branch":\s*"[^"]*",\s*"unit_branch_source":\s*"[^"]*",?[ \t]*$',
    re.MULTILINE,
)


def fixture_t1_card_schema_line_substituted_binds() -> None:
    """For EACH of the 7 verifier cards: scrape the card's OWN literal
    schema line, substitute a real branch/source into it exactly as a
    verifier filling in the template would, and confirm the resulting
    verdict BINDS — files under the real branch, never UNPARSEABLE
    (template), never UNBOUND. For the 3 cards the PreToolUse gate actually
    re-dispatches on (`_VERIFIER_SECOND_ROUND`), also confirm a second
    dispatch naming that exact branch is denied — proof this is a REAL
    bind, not merely a stored string."""
    for name in _CARD_NAMES:
        text = (AGENTS_DIR / f"{name}.md").read_text()
        m = _CARD_UNIT_BRANCH_LINE_RE.search(text)
        _assert(m is not None, "T1",
                f"{name}.md must carry the bare-branch unit_branch schema line "
                "(\"unit_branch\": \"...\", \"unit_branch_source\": \"...\",)")
        real_branch = f"feat/tcard-{name}"
        line = m.group(0)
        substituted = re.sub(r'"unit_branch":\s*"[^"]*"', f'"unit_branch": "{real_branch}"', line)
        substituted = re.sub(r'"unit_branch_source":\s*"[^"]*"', '"unit_branch_source": "git"', substituted)
        substituted = substituted.strip()
        if substituted.endswith(","):
            substituted = substituted[:-1]
        msg = ("```json\n{\"kind\": \"verdict\", \"verdict\": \"BLOCK\", "
               + substituted + ", \"findings\": []}\n```")
        root = _fresh_root()
        row = _stop_and_read(root, msg, agent_type=name)
        _assert(row is not None, "T1", f"[{name}] expected a row")
        _assert(row["verdict"] != "UNPARSEABLE", "T1",
                f"[{name}] a schema-line-substituted real branch must NOT be treated as a "
                f"template, got verdict={row['verdict']!r} reason={row.get('unparseable_reason')!r}")
        _assert(row.get("unit_branch") == real_branch, "T1",
                f"[{name}] unit_branch must bind to the bare branch, got {row.get('unit_branch')!r}")
        if name in _VERIFIER_SECOND_ROUND:
            p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
                "subagent_type": name, "prompt": f"re-check unit: {real_branch}"}}, root)
            _assert(p.returncode == 2, "T1",
                    f"[{name}] a second dispatch naming the bound unit_branch must be denied, "
                    f"got {p.returncode}: {p.stderr}")


def fixture_t2_annotated_legacy_unit_branch_binds() -> None:
    """Backward compatibility with in-flight transcripts: an annotated
    legacy-shape `unit_branch` — `"<branch> (from git)"`, the pre-fix
    producer output every card's OLD "say which" wording actually elicited
    — must ALSO bind via leading-token normalization, with the parenthetical
    preserved as `unit_branch_note`, never dropped silently and never
    causing a template misclassification."""
    root = _fresh_root()
    real_branch = "feat/t2-legacy"
    v = {"kind": "verdict", "verdict": "BLOCK", "unit_branch": f"{real_branch} (from git)", "findings": []}
    msg = "```json\n" + json.dumps(v) + "\n```"
    row = _stop_and_read(root, msg, agent_type="adversarial-audit")
    _assert(row is not None, "T2", "expected a row")
    _assert(row["verdict"] != "UNPARSEABLE", "T2",
            f"an annotated-but-real unit_branch must NOT be treated as a template, got "
            f"{row['verdict']!r} ({row.get('unparseable_reason')!r})")
    _assert(row.get("unit_branch") == real_branch, "T2",
            f"leading-token normalization must strip the parenthetical, got {row.get('unit_branch')!r}")
    _assert(row.get("unit_branch_note") == "(from git)", "T2",
            f"the annotation must be preserved as a note, got {row.get('unit_branch_note')!r}")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {real_branch}"}}, root)
    _assert(p.returncode == 2, "T2",
            f"the normalized branch must gate a second dispatch, got {p.returncode}: {p.stderr}")


def fixture_t3_start_binds_unit_branch_colon_form() -> None:
    """`_UNIT_LINE_RE` (`^unit:`) bound 0/126 real Starts — the lead's real
    dispatch prompts carry `unit_branch:` instead. This is the SubagentStart
    advisory binding (used only to file an UNPARSEABLE verdict under the
    right unit), not a gate decision."""
    root = _fresh_root()
    p = _run("lead-gate-start.sh", {"agent_id": "t3", "agent_type": "adversarial-audit",
                                     "prompt": "Dispatch details.\nunit_branch: feat/t3-colon\nGo audit it."}, root)
    _assert(p.returncode == 0, "T3 setup", "start must exit 0")
    p = _run("lead-gate-stop.sh", {"agent_id": "t3", "agent_type": "adversarial-audit",
                                    "last_assistant_message": "ran out of context, no verdict"}, root)
    _assert(p.returncode == 0, "T3 setup", "stop must exit 0")
    f = root / ".jammi" / "gate-state" / "feat_t3-colon.jsonl"
    present = sorted(p.name for p in (root / ".jammi" / "gate-state").iterdir()) if (root / ".jammi" / "gate-state").exists() else []
    _assert(f.exists(), "T3",
            f"expected the UNPARSEABLE row filed under feat_t3-colon (the 'unit_branch:' colon-form "
            f"binding), files present: {present}")
    row = json.loads(f.read_text().splitlines()[-1])
    _assert(row["verdict"] == "UNPARSEABLE", "T3", f"expected UNPARSEABLE, got {row['verdict']!r}")


def fixture_t4_start_binds_unit_branch_bare_form() -> None:
    """The bare (no-colon) `unit_branch <value>` shape — the second real
    dispatch shape the lead writes — also binds the SubagentStart."""
    root = _fresh_root()
    p = _run("lead-gate-start.sh", {"agent_id": "t4", "agent_type": "adversarial-audit",
                                     "prompt": "Dispatch details.\nunit_branch feat/t4-bare\nGo audit it."}, root)
    _assert(p.returncode == 0, "T4 setup", "start must exit 0")
    p = _run("lead-gate-stop.sh", {"agent_id": "t4", "agent_type": "adversarial-audit",
                                    "last_assistant_message": "ran out of context, no verdict"}, root)
    _assert(p.returncode == 0, "T4 setup", "stop must exit 0")
    f = root / ".jammi" / "gate-state" / "feat_t4-bare.jsonl"
    present = sorted(p.name for p in (root / ".jammi" / "gate-state").iterdir()) if (root / ".jammi" / "gate-state").exists() else []
    _assert(f.exists(), "T4",
            f"expected the UNPARSEABLE row filed under feat_t4-bare (the bare 'unit_branch <value>' "
            f"binding), files present: {present}")
    row = json.loads(f.read_text().splitlines()[-1])
    _assert(row["verdict"] == "UNPARSEABLE", "T4", f"expected UNPARSEABLE, got {row['verdict']!r}")


# ==========================================================================
# L1 / L2 — the closed-world agent-type lattice
# ==========================================================================

def fixture_l1_unknown_type_denied() -> None:
    root = _fresh_root()
    _write_block_row(root, "feat/l1", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    for t in ("lead", "some-future-agent-type", "Bash"):
        p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": t, "prompt": "unit: feat/l1\nfix a.py:1"}}, root)
        _assert(p.returncode == 2, "L1", f"unrecognized type {t!r} must deny, got {p.returncode}")
        _assert("unknown agent type" in p.stderr, "L1", f"reason must say 'unknown agent type': {p.stderr!r}")


def _lib_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("lead_gate_lib_l2", LEAD_GATE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def fixture_l2_agent_card_lattice_cross_check() -> None:
    mod = _lib_module()
    gated, never_gated = mod.GATED_TYPES, mod.NEVER_GATED_TYPES
    universe = set(mod.HARNESS_BUILTIN_TYPES)
    tools_by_name: dict[str, list[str]] = {}
    for path in sorted(AGENTS_DIR.glob("*.md")):
        text = path.read_text()
        nm = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
        tl = re.search(r"^tools:\s*\[([^\]]*)\]", text, re.MULTILINE)
        if not nm:
            continue
        name = nm.group(1)
        if name == "lead":
            continue
        universe.add(name)
        if tl:
            tools_by_name[name] = [t.strip() for t in tl.group(1).split(",")]

    unclassified = [n for n in universe if n not in gated and n not in never_gated]
    _assert(not unclassified, "L2",
            f"agent type(s) in neither GATED_TYPES nor NEVER_GATED_TYPES: {unclassified}")

    write_tools = {"Edit", "Write", "MultiEdit"}
    for name, tools in tools_by_name.items():
        if name in never_gated:
            leaked = write_tools & set(tools)
            _assert(not leaked, "L2",
                    f"NEVER_GATED {name!r} declares write tool(s) {leaked} in its tools: frontmatter")


def fixture_l3_subtype_key_spellings_and_distinct_absent_arm() -> None:
    """audit-r3 finding 4: a payload without `subagent_type` (or spelling
    the field differently) collapsed into deny-unknown with the
    unrepresentable remedy "add '' to GATED_TYPES". The field is read under
    every known spelling, and the absent-field case is its own deny arm
    with its own remedy."""
    root = _fresh_root()
    # Alternate spelling behaves exactly like `subagent_type` (allow path).
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "agent_type": "adversarial-audit", "prompt": "unit: feat/l3\nfirst audit"}}, root)
    _assert(p.returncode == 0, "L3", f"'agent_type' spelling must be read, got {p.returncode}: {p.stderr}")
    # Alternate spelling behaves exactly like `subagent_type` (deny-unknown path).
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "agent_type": "lead", "prompt": "unit: feat/l3\nx"}}, root)
    _assert(p.returncode == 2, "L3", f"an unknown type under an alternate spelling must deny, got {p.returncode}")
    _assert("unknown agent type" in p.stderr, "L3", f"reason must be the unknown-type arm: {p.stderr!r}")
    # NO agent-type field at all: a DISTINCT deny arm, never the unknown-type arm.
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "prompt": "unit: feat/l3\nx"}}, root)
    _assert(p.returncode == 2, "L3", f"a payload with no agent-type field must deny, got {p.returncode}")
    _assert("no agent-type field" in p.stderr, "L3",
            f"absent-field deny must be its own arm with its own remedy: {p.stderr!r}")
    _assert("unknown agent type" not in p.stderr, "L3",
            f"absent-field deny must NOT collapse into the unknown-type arm: {p.stderr!r}")


# ==========================================================================
# V1-V10 — verdict parsing / row validity
# ==========================================================================

def _stop_and_read(root: Path, msg: str, agent_type: str = "adversarial-audit",
                    agent_id: str = "a") -> dict | None:
    p = _run("lead-gate-stop.sh", {"agent_id": agent_id, "agent_type": agent_type,
                                    "last_assistant_message": msg}, root)
    _assert(p.returncode == 0, "verdict-row setup", f"stop must exit 0, got {p.returncode}")
    d = root / ".jammi" / "gate-state"
    for f in d.iterdir():
        if f.name not in ("bindings.jsonl", "hook.log") and f.suffix == ".jsonl" and ".relay." not in f.name:
            return json.loads(f.read_text().splitlines()[-1])
    return None


def fixture_v1_template_after_real_is_unparseable() -> None:
    root = _fresh_root()
    real = {"kind": "verdict", "verdict": "BLOCK", "unit_branch": "feat/x",
            "class_enumeration": ["a.py:1"], "sweep_method": "g", "exhaustive": True, "findings": []}
    tmpl = {"kind": "verdict", "verdict": "BLOCK | PASS", "unit_branch": "<branch>",
            "class_enumeration": ["path:line"]}
    msg = ("```json\n" + json.dumps(real) + "\n```\nFor reference the schema is:\n```json\n"
           + json.dumps(tmpl) + "\n```")
    row = _stop_and_read(root, msg)
    _assert(row is not None, "V1", "expected a row")
    _assert(row["verdict"] == "UNPARSEABLE", "V1", f"a template echoed LAST must be UNPARSEABLE, got {row['verdict']!r}")
    _assert(row.get("unparseable_reason") == "template", "V1", f"reason must be 'template', got {row.get('unparseable_reason')!r}")


def fixture_v2_truncated_writes_unparseable_row() -> None:
    root = _fresh_root()
    real = {"kind": "verdict", "verdict": "BLOCK", "unit_branch": "feat/x",
            "class_enumeration": ["a.py:1"], "findings": []}
    msg = "```json\n" + json.dumps(real)[:20]  # truncated mid-object, no closing brace
    row = _stop_and_read(root, msg)
    _assert(row is not None, "V2", "a truncated block MUST still write a row — none was written")
    _assert(row["verdict"] == "UNPARSEABLE", "V2", f"expected UNPARSEABLE, got {row['verdict']!r}")


def fixture_v3_unrecognized_value_diagnosable() -> None:
    root = _fresh_root()
    v = {"kind": "verdict", "verdict": "PASS (no HARD_BLOCK)", "unit_branch": "feat/v3",
         "class_enumeration": [], "findings": []}
    msg = "```json\n" + json.dumps(v) + "\n```"
    row = _stop_and_read(root, msg)
    _assert(row["verdict"] == "BLOCK", "V3", f"unrecognized value must default-BLOCK, got {row['verdict']!r}")
    _assert(row.get("verdict_raw") == "PASS (no HARD_BLOCK)", "V3", "raw value must be preserved")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "unit: feat/v3\nunrelated"}}, root)
    _assert(p.returncode == 2, "V3", "expected a deny")
    _assert("PASS (no HARD_BLOCK)" in p.stderr, "V3", f"deny reason must name the raw value: {p.stderr!r}")


def fixture_v4_v5_v6_pass_like_pinned() -> None:
    for raw, agent_type in (("PASS", "adversarial-audit"), ("verified", "fix-verifier"),
                             ("PROCEED", "pressure-tester")):
        root = _fresh_root()
        v = {"kind": "verdict", "verdict": raw, "unit_branch": f"feat/{raw}", "findings": []}
        msg = "```json\n" + json.dumps(v) + "\n```"
        row = _stop_and_read(root, msg, agent_type=agent_type)
        _assert(row["verdict"] == "PASS", f"V4-6[{raw}]", f"{raw!r} must normalize to PASS, got {row['verdict']!r}")


def fixture_v7_tag_inside_notes_string_does_not_corrupt() -> None:
    """round-2 finding 6: `</verdict>` inside the verdict's own `notes`
    string must not truncate the region — exercised via the LEGACY TAG
    fallback (no fenced ```json block at all), which is where the bug
    lived."""
    root = _fresh_root()
    real = {
        "verdict": "BLOCK", "unit_branch": "ci/real", "head_sha": "b1b1828",
        "worktree": "/w/X", "class_enumeration": ["a.py:1", "b.py:2", "c.py:3"],
        "findings": [],
        "notes": "Parser: a template placed after the real block misparses; the closing </verdict> tag must be last.",
    }
    msg = "<verdict>\n" + json.dumps(real) + "\n</verdict>"
    row = _stop_and_read(root, msg)
    _assert(row is not None, "V7", "expected a row")
    _assert(row["unit_branch"] == "ci/real", "V7", f"the notes-embedded tag must not corrupt the real BLOCK, got unit_branch={row.get('unit_branch')!r}")
    _assert(row["verdict"] == "BLOCK", "V7", f"expected BLOCK, got {row['verdict']!r}")
    _assert(row["class_enumeration"] == ["a.py:1", "b.py:2", "c.py:3"], "V7", "sites must survive intact")


def fixture_v8_unparseable_filed_under_start_binding() -> None:
    root = _fresh_root()
    p = _run("lead-gate-start.sh", {"agent_id": "a9", "agent_type": "adversarial-audit",
                                     "prompt": "unit: feat/v8\naudit please"}, root)
    _assert(p.returncode == 0, "V8 setup", "start must exit 0")
    p = _run("lead-gate-stop.sh", {"agent_id": "a9", "agent_type": "adversarial-audit",
                                    "last_assistant_message": "I could not produce a verdict."}, root)
    _assert(p.returncode == 0, "V8 setup", "stop must exit 0")
    f = root / ".jammi" / "gate-state" / "feat_v8.jsonl"
    _assert(f.exists(), "V8", "UNPARSEABLE row must be filed under the start binding (feat_v8), not UNBOUND")
    row = json.loads(f.read_text().splitlines()[-1])
    _assert(row["verdict"] == "UNPARSEABLE", "V8", f"expected UNPARSEABLE, got {row['verdict']!r}")
    unbound_f = root / ".jammi" / "gate-state" / "UNBOUND.jsonl"
    _assert(not unbound_f.exists(), "V8", "must NOT also file under UNBOUND when a binding exists")


def fixture_v9_wrong_kind_is_unparseable() -> None:
    root = _fresh_root()
    v = {"kind": "not-a-verdict", "verdict": "PASS", "unit_branch": "feat/v9"}
    msg = "```json\n" + json.dumps(v) + "\n```"
    row = _stop_and_read(root, msg)
    _assert(row["verdict"] == "UNPARSEABLE", "V9",
            f"a fenced block with the wrong kind must be UNPARSEABLE (no fallback to the tag walk), got {row['verdict']!r}")


def fixture_v10_marker_inside_notes_string_is_ignored() -> None:
    """audit-r3 finding 2: the OPENING fence-marker scan was not
    string-aware, so a PASS whose `notes` mentioned the marker was recorded
    UNPARSEABLE and gated the unit closed. The scan jumps past every parsed
    object before looking for a later marker."""
    root = _fresh_root()
    v = {"kind": "verdict", "verdict": "PASS", "unit_branch": "feat/v10",
         "class_enumeration": [], "findings": [],
         "notes": "the card's own ```json fence template was followed exactly"}
    msg = "All clear.\n```json\n" + json.dumps(v) + "\n```\nDone."
    row = _stop_and_read(root, msg)
    _assert(row is not None, "V10", "expected a row")
    _assert(row["verdict"] == "PASS", "V10",
            f"a marker inside the verdict's own notes STRING must not corrupt the parse, got {row['verdict']!r}")
    _assert(row["unit_branch"] == "feat/v10", "V10", f"got unit_branch={row.get('unit_branch')!r}")


# ==========================================================================
# E1-E4 — exit lattice
# ==========================================================================

def fixture_e1_non_utf8_payload() -> None:
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", b"\xff\xfe{not utf-8 garbage", root)
    _assert(p.returncode == 2, "E1", f"a non-UTF-8 payload must exit 2 (never 1), got {p.returncode}")


def fixture_e2_missing_python3() -> None:
    root = _fresh_root()
    empty_bin = root / "empty-bin"
    empty_bin.mkdir()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {}}, root,
             env_overrides={"PATH": str(empty_bin)})
    _assert(p.returncode == 2, "E2", f"a missing python3 must exit 2 (never 1), got {p.returncode}")


def fixture_e4_json_invalid_payload_fails_closed() -> None:
    """audit-r3 finding 3: a UTF-8-valid but JSON-invalid payload was
    caught into `{}` and ALLOWED (only the decode-error sibling exited 2).
    `pre` must exit 2 on a JSON-invalid payload AND on a JSON non-object;
    `stop` stays a best-effort writer (exit 0, no row)."""
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", b'{"tool_name": "Agent", "tool_input": {broken', root)
    _assert(p.returncode == 2, "E4", f"a JSON-invalid payload must exit 2 (never allow), got {p.returncode}")
    p = _run("lead-gate-pre.sh", b'[1, 2, 3]', root)
    _assert(p.returncode == 2, "E4", f"a JSON non-object payload must exit 2, got {p.returncode}")
    p = _run("lead-gate-pre.sh", b'', root)
    _assert(p.returncode == 2, "E4", f"an empty payload must exit 2, got {p.returncode}")
    p = _run("lead-gate-stop.sh", b'{"broken', root)
    _assert(p.returncode == 0, "E4", f"stop is a best-effort writer, must exit 0, got {p.returncode}")


def fixture_e3_broken_python3_on_path() -> None:
    root = _fresh_root()
    fake_bin = root / "fake-bin"
    fake_bin.mkdir()
    fake_python3 = fake_bin / "python3"
    fake_python3.write_text("#!/bin/sh\nexit 7\n")
    fake_python3.chmod(0o755)
    real_path = os.environ.get("PATH", "")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {}}, root,
             env_overrides={"PATH": f"{fake_bin}:{real_path}"})
    _assert(p.returncode == 2, "E3", f"a python3 exiting 7 must be mapped to 2, got {p.returncode}")


# ==========================================================================
# S1-S3 / D1 — "only verifier stops write rows" (capability commit 2).
# ==========================================================================

def fixture_s1_non_verifier_stop_writes_nothing() -> None:
    """`STOP_MATCH_TYPES` wired into `handle_stop`: a stop payload whose
    agent_type is NOT a verifier type (here, an empty string — the settings.
    json matcher's own defense-in-depth backstop) writes NO `.jsonl` row —
    no unit row, no UNBOUND append — even though `stop` still exits 0
    (never denies)."""
    root = _fresh_root()
    msg = "```json\n" + json.dumps(
        {"kind": "verdict", "verdict": "BLOCK", "unit_branch": "feat/s1", "findings": []}
    ) + "\n```"
    p = _run("lead-gate-stop.sh", {"agent_id": "s1", "agent_type": "",
                                    "last_assistant_message": msg}, root)
    _assert(p.returncode == 0, "S1", f"stop must still exit 0, got {p.returncode}")
    gate_dir = root / ".jammi" / "gate-state"
    jsonl_files = list(gate_dir.glob("*.jsonl")) if gate_dir.exists() else []
    _assert(not jsonl_files, "S1",
            f"a non-verifier stop must write NO .jsonl row (no UNBOUND append either), "
            f"found: {[f.name for f in jsonl_files]}")


def fixture_s2_verifier_stop_still_writes() -> None:
    """A verifier-typed stop (drawn from STOP_MATCH_TYPES, exercised here
    with `oracle` — the member least covered by the other write-path
    fixtures) still writes its row."""
    root = _fresh_root()
    msg = "```json\n" + json.dumps(
        {"kind": "verdict", "verdict": "HARD_BLOCK", "unit_branch": "feat/s2", "findings": []}
    ) + "\n```"
    p = _run("lead-gate-stop.sh", {"agent_id": "s2", "agent_type": "oracle",
                                    "last_assistant_message": msg}, root)
    _assert(p.returncode == 0, "S2", f"stop must exit 0, got {p.returncode}")
    f = root / ".jammi" / "gate-state" / "feat_s2.jsonl"
    _assert(f.exists(), "S2", "a verifier-typed stop (oracle) must still write its row")


def fixture_s3_unbound_rotation_at_cap() -> None:
    """A pre-existing oversized UNBOUND.jsonl (carrying an open BLOCK row)
    is rotated to `UNBOUND.jsonl.1` on the next `stop` invocation — BEFORE
    the STOP_MATCH_TYPES filter runs (a verifier-typed stop triggers
    rotation just like any other). `all_open_blocks` must no longer see
    the rotated content (its `entry.suffix != ".jsonl"` check already
    excludes it — no further gate-side change needed)."""
    root = _fresh_root()
    sdir = root / ".jammi" / "gate-state"
    sdir.mkdir(parents=True, exist_ok=True)
    mod = _lib_module()
    cap = mod._UNBOUND_ROTATE_CAP_BYTES
    unbound = sdir / "UNBOUND.jsonl"
    row_line = json.dumps({
        "ts": "2020-01-01T00:00:00Z", "agent_id": "x", "agent_type": "adversarial-audit",
        "unit_branch": "UNBOUND", "unit_branch_note": None, "head_sha": None, "worktree": None,
        "verdict": "BLOCK", "verdict_raw": "BLOCK", "unparseable_reason": None,
        "round": 1, "class_enumeration": [], "enumeration_missing": True,
        "sweep_method": None, "exhaustive": False, "finding_locations": [],
        "pad": "x" * 2000,
    }) + "\n"
    with unbound.open("wb") as f:
        while f.tell() <= cap:
            f.write(row_line.encode("utf-8"))
    original_size = unbound.stat().st_size
    _assert(original_size > cap, "S3 setup", f"pre-seeded file must exceed the cap ({cap}), got {original_size}")

    msg = "```json\n" + json.dumps(
        {"kind": "verdict", "verdict": "PASS", "unit_branch": "feat/s3", "findings": []}
    ) + "\n```"
    p = _run("lead-gate-stop.sh", {"agent_id": "s3", "agent_type": "adversarial-audit",
                                    "last_assistant_message": msg}, root)
    _assert(p.returncode == 0, "S3", f"stop must exit 0, got {p.returncode}")

    rotated = sdir / "UNBOUND.jsonl.1"
    _assert(rotated.exists(), "S3", "the oversized UNBOUND.jsonl must be rotated to UNBOUND.jsonl.1")
    _assert(rotated.stat().st_size == original_size, "S3",
            "the rotated sibling must carry the pre-rotation content byte-for-byte")
    _assert(not unbound.exists() or unbound.stat().st_size < cap, "S3",
            "the live UNBOUND.jsonl must not still carry the oversized content after rotation")

    opens = mod.all_open_blocks(sdir)
    _assert(not any(u == "UNBOUND" for u, _t, _r, _i in opens), "S3",
            f"all_open_blocks must not re-parse the rotated UNBOUND.jsonl.1, got: {opens}")


def fixture_d1_recognized_block_not_mislabeled_unrecognized() -> None:
    """Cosmetic `_diagnose_row` fix: the RECOGNIZED literal `"BLOCK"` value
    (the `BLOCK | PASS` vocabulary's own spelling — adversarial-audit,
    citation-checker, discipline-test-auditor) must never be reported as
    "unrecognized verdict value ... defaulted to BLOCK" in a second-round
    deny reason; a genuinely unrecognized value (exercised elsewhere, V3)
    still is."""
    root = _fresh_root()
    row = _write_block_row(root, "feat/d1", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _assert(row.get("verdict_raw") == "BLOCK", "D1 setup", f"expected verdict_raw 'BLOCK', got {row.get('verdict_raw')!r}")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {row['worktree']}"}}, root)
    _assert(p.returncode == 2, "D1", f"expected a deny, got {p.returncode}")
    _assert("unrecognized verdict value" not in p.stderr, "D1",
            f"the recognized 'BLOCK' spelling must not be mislabeled 'unrecognized': {p.stderr!r}")


# ==========================================================================
# N6 / N7 — must-still-count
# ==========================================================================

def fixture_n6_nothing_gated_when_no_block_anywhere() -> None:
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent",
                                   "tool_input": {"subagent_type": "adversarial-audit",
                                                   "prompt": "no BLOCK anywhere"}}, root)
    _assert(p.returncode == 0, "N6", f"expected allow with no state at all, got {p.returncode}: {p.stderr}")


def fixture_r10_wiring() -> None:
    _assert(SETTINGS_PATH.exists(), "R10", f"{SETTINGS_PATH} must exist")
    settings = json.loads(SETTINGS_PATH.read_text())
    hooks = settings.get("hooks", {})

    def _script_names(event: str) -> set[str]:
        names: set[str] = set()
        for entry in hooks.get(event, []):
            for h in entry.get("hooks", []):
                cmd = h.get("command", "")
                names.add(Path(cmd.split()[-1] if " " in cmd else cmd).name)
        return names

    _assert("lead-gate-start.sh" in _script_names("SubagentStart"), "R10", "SubagentStart must wire lead-gate-start.sh")
    _assert("lead-gate-stop.sh" in _script_names("SubagentStop"), "R10", "SubagentStop must wire lead-gate-stop.sh")
    _assert("lead-gate-pre.sh" in _script_names("PreToolUse"), "R10", "PreToolUse must wire lead-gate-pre.sh")

    pre_matchers = "|".join(e.get("matcher", "") for e in hooks.get("PreToolUse", []))
    for tok in ("Agent", "Task"):
        _assert(tok in pre_matchers, "R10", f"PreToolUse matcher set must cover {tok!r}: {pre_matchers!r}")

    for name in ("lead-gate-start.sh", "lead-gate-stop.sh", "lead-gate-pre.sh", "lead-gate-lib.py"):
        path = HOOKS_DIR / name
        _assert(path.exists(), "R10", f"{path} must exist")
        _assert(os.access(path, os.X_OK), "R10", f"{path} must be executable")

    deny = settings.get("permissions", {}).get("deny", [])
    deny_text = "\n".join(deny)
    _assert(".claude/hooks/" in deny_text, "R10", f"permissions.deny must cover .claude/hooks/**: {deny}")
    _assert(".claude/settings.json" in deny_text, "R10", f"permissions.deny must cover .claude/settings.json: {deny}")


FIXTURES = [
    ("G1", fixture_g1_first_round_never_gated),
    ("G2", fixture_g2_second_round_denied_worktree),
    ("G3", fixture_g3_second_round_denied_full_sha),
    ("G4", fixture_g4_second_round_denied_short_sha),
    ("G5", fixture_g5_second_round_denied_unit_branch),
    ("G6", fixture_g6_second_round_allowed_with_accepted_relay),
    ("G7", fixture_g7_relay_missing_site_not_accepted),
    ("G8", fixture_g8_enumeration_missing_probe_fallback),
    ("G9", fixture_g9_mismatched_relay_not_accepted),
    ("G10", fixture_g10_dodge5_unlabeled_redispatch_allowed),
    ("G11", fixture_g11_cross_type_non_interference),
    ("G12", fixture_g12_same_type_pass_clears),
    ("G13", fixture_g13_verifier_pass_clears_audited_block_only_with_relay),
    ("G14", fixture_g14_unparseable_row_gates_like_block),
    ("G15", fixture_g15_whole_token_anchors_never_raw_substrings),
    ("T1", fixture_t1_card_schema_line_substituted_binds),
    ("T2", fixture_t2_annotated_legacy_unit_branch_binds),
    ("T3", fixture_t3_start_binds_unit_branch_colon_form),
    ("T4", fixture_t4_start_binds_unit_branch_bare_form),
    ("L1", fixture_l1_unknown_type_denied),
    ("L2", fixture_l2_agent_card_lattice_cross_check),
    ("L3", fixture_l3_subtype_key_spellings_and_distinct_absent_arm),
    ("V1", fixture_v1_template_after_real_is_unparseable),
    ("V2", fixture_v2_truncated_writes_unparseable_row),
    ("V3", fixture_v3_unrecognized_value_diagnosable),
    ("V4-6", fixture_v4_v5_v6_pass_like_pinned),
    ("V7", fixture_v7_tag_inside_notes_string_does_not_corrupt),
    ("V8", fixture_v8_unparseable_filed_under_start_binding),
    ("V9", fixture_v9_wrong_kind_is_unparseable),
    ("V10", fixture_v10_marker_inside_notes_string_is_ignored),
    ("E1", fixture_e1_non_utf8_payload),
    ("E2", fixture_e2_missing_python3),
    ("E3", fixture_e3_broken_python3_on_path),
    ("E4", fixture_e4_json_invalid_payload_fails_closed),
    ("S1", fixture_s1_non_verifier_stop_writes_nothing),
    ("S2", fixture_s2_verifier_stop_still_writes),
    ("S3", fixture_s3_unbound_rotation_at_cap),
    ("D1", fixture_d1_recognized_block_not_mislabeled_unrecognized),
    ("N6", fixture_n6_nothing_gated_when_no_block_anywhere),
    ("R10", fixture_r10_wiring),
]


def self_test() -> int:
    failures: list[str] = []
    for name, fn in FIXTURES:
        _WALL_TIMES.clear()
        try:
            fn()
            print(f"check-lead-gate[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-lead-gate[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-lead-gate[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)

    root = _fresh_root()
    times = []
    for _ in range(5):
        start = time.monotonic()
        _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": "adversarial-audit", "prompt": "unit: x\nfoo"}}, root)
        times.append(time.monotonic() - start)
    slow = [t for t in times if t >= 1.0]
    if slow:
        failures.append(f"N7: {len(slow)}/{len(times)} invocation(s) took >= 1s: {slow}")
        print(f"check-lead-gate[N7]: FAIL — {slow}", file=sys.stderr)
    else:
        print(f"check-lead-gate[N7]: OK (max {max(times):.3f}s over {len(times)} invocations)")

    if failures:
        print("check-lead-gate: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"check-lead-gate: all {len(FIXTURES) + 1} self-test fixture(s) passed.")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    print("check_lead_gate.py: usage: --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
