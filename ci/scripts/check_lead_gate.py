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
  G9  a relay artifact with a mismatched block_ts/agent_type is NOT
      accepted; a mismatched `unit_branch` is NOT itself an R1/R2 identity
      check any more (esc-097 V16 moved that binding into R3's own,
      git-free reachability check — see G22b/G31)
  G10 DODGE-5: an unlabeled re-dispatch (names none of the 3 anchors) is
      allowed — the documented residual
  G11 cross-type non-interference: a pressure-tester REFINE does not gate
      the first adversarial-audit dispatch (built non-vacuously)
  G12 a SAME-agent_type PASS clears its own BLOCK
  G13 esc-097 (V10): there is no cross-type clearing arm at all -- a
      fix-verifier PASS is irrelevant to an adversarial-audit BLOCK's own
      repeat dispatch; G13a (no relay) still denies, G13b (a relay accepted
      under the audit's OWN same-type R1+R2+R3 rule) allows regardless of
      an unrelated fix-verifier PASS on record
  G14 an UNPARSEABLE latest row gates exactly like a BLOCK (`is_open`
      covers both values — audit-r3 finding 5's surviving mutant)
  G15 anchors bind as WHOLE TOKENS, never raw substrings (audit-r3 finding
      1): an open BLOCK on `ci/gpu` does NOT gate `ci/gpu-dev`, `<worktree>2`
      and a 7-char-lookalike hex token do NOT gate; a path UNDER the
      recorded worktree and a TRUE >=7-char sha prefix still DO
  G16 esc-064 RED: a BLOCK with a NON-EMPTY class_enumeration whose relay
      restates it as `sites` with NO `probe` is NOT accepted -> deny(2),
      and the deny reason NAMES the missing probe evidence
  G17 GREEN: the same relay plus >=2 disjoint probe sites IS accepted
  G18 probe boundaries (each root isolates ONE axis, all deny): collision
      with an ENUMERATED-not-merely-found site; count 1; empty/whitespace
      entries; duplicates; whitespace-padded collision; strip-identical
      pair counting as one; a VERIFIER-emitted padded finding location
      restated unpadded; zero-width (Cf) invisible-character collision
  G19 the coverage arm is selected by the DATA: a row with a non-empty
      class_enumeration but NO `enumeration_missing` key still requires
      `sites` (the flag is diagnostic, never a discriminator)
  OQ1/OQ2  esc-lead-gate-R10: alongside the >=2 examined-clean probe sites
      R2 already requires, a relay must ALSO carry a non-empty
      `open_question` — a site examined and explicitly NOT closed, naming
      the attack for the next round. OQ1: an otherwise-fully-satisfying
      relay (R1/R2/R3 all pass) with NO `open_question` is denied, naming
      the missing field. OQ2: the SAME relay plus a non-empty
      `open_question` is allowed. Armed unconditionally, alongside R2 —
      never folded into `probe`'s own >=2 count.
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
  V4/V5/V6  the per-agent-type PASS vocabulary (esc-lead-gate-R7d) is pinned
      individually: "PASS" (adversarial-audit), "verified" (fix-verifier),
      "PROCEED" (pressure-tester) each clear THEIR OWN card's row; three
      negatives pin the fix itself — adversarial-audit "verified",
      adversarial-audit "PROCEED", and oracle "verified" must NOT clear,
      because none is that card's own spelling (the pooled global set used
      to let all three clear)
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
  G20-G40 esc-097 (R3, "probe the fix" — proposal, docs/plans/63-how-well/
      proposals/esc-097-probe-the-fix.md): exercised behind a `RELAY_R3`
      version-marker guard on `.claude/hooks/lead-gate-lib.py` — reported
      SKIPPED (exit 0 for that arm only) until a human applies the
      proposal's patch files; run for real, and must all pass, the moment
      the marker is present. G32-G35 are the round-2 pressure-test
      reproducers (V10-V17): G32 proves the deleted cross-type clearing
      arm stays dead (a relay with no `fix_head`, plus a fix-verifier AND
      an acceptance-verifier PASS, still denies a repeat adversarial-audit
      dispatch); G33/G34 prove a prompt naming MORE THAN ONE open BLOCK of
      the same type denies outright, naming both, in each name order;
      G35 proves the argv boundary (a `head_sha` shaped like a git option
      denies and spawns no git that could act on it). G36-G38 are the
      round-3 adversarial reproducers (V18): G25 (rewritten) and G36 both
      DENY a `fix_head` that resolves as a real object but is reachable
      from no ref (an amend abandons it); G37 DENIES a `fix_head` that is
      real and ref-reachable but lives on an UNRELATED branch; all three
      RED against the 7633b2d6 patch (which trusted the relay's
      `unit_branch` NAME alone and never checked `fix_head`'s POSITION)
      and GREEN once V18's `git merge-base --is-ancestor` check lands. G38
      proves the git subprocess's own stderr text is read and appended to
      the deny reason, not merely a bare exit code. G39/G40 are the
      round-4 adversarial reproducers: G39 DENIES a TAG literally named
      like `unit_branch`, pointing at another branch's commit, that would
      otherwise shadow the real branch's tip via gitrevisions(7)'s own
      refs/tags-before-refs/heads disambiguation (RED against 7ed0db7d's
      patch, which resolved `unit_branch` as a bare `<name>^{commit}` and
      ALLOWED this exact shape); G40 DENIES a BRANCH literally named like
      `fix_head`'s own hex prefix, pointing elsewhere, that would otherwise
      shadow the abbreviated object (RED against 7ed0db7d's patch, which
      never checked that a resolved sha STARTS WITH the hex given). Both
      are GREEN once `unit_branch` resolves under `refs/heads/` only and
      `block_sha`/`fix_head` are resolved via `git rev-parse --verify` with
      a startswith check on the result.

Run: `python3 ci/scripts/check_lead_gate.py --self-test`
"""

from __future__ import annotations

import atexit
import hashlib
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
         env_overrides: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """esc-097 (HARD_BLOCK fix): `cwd` is NOT `project_dir` by default — a
    mutant that reduces `repo_root()` to `Path.cwd()` (dropping the
    `CLAUDE_PROJECT_DIR` env read entirely) under the 8b4e4b9d harness
    passed all 45 fixtures here BECAUSE every one of them ran with
    `cwd == project_dir`,
    so cwd-vs-env was never actually distinguished by anything. The default
    `cwd` is now a single, shared, empty DECOY directory (`_DECOY_CWD`,
    minted once at import time) that carries no `.jammi/gate-state` at all —
    a fixture that needs the cwd FALLBACK exercised on purpose (only G26:
    `CLAUDE_PROJECT_DIR` unset, relying on `repo_root()`'s OTHER, unrelated
    cwd fallback for the non-R3 arms) passes `cwd=project_dir` explicitly."""
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    if env_overrides:
        # Applied AFTER the default CLAUDE_PROJECT_DIR so a caller can
        # override it (e.g. a PATH shim) or UNSET it entirely (a value of
        # `None` pops the key — G26 exercises exactly this: an unset
        # CLAUDE_PROJECT_DIR is its own deny arm, and it must be reachable
        # from this harness, not merely from the lib in isolation).
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    script_path = HOOKS_DIR / script
    data = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload).encode("utf-8")
    start = time.monotonic()
    proc = subprocess.run(
        ["/bin/sh", str(script_path)],
        input=data,
        capture_output=True,
        env=env,
        cwd=str(cwd if cwd is not None else _DECOY_CWD),
        timeout=10,
    )
    _WALL_TIMES.append(time.monotonic() - start)
    proc.stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else proc.stdout
    proc.stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
    return proc


# Every tempdir this harness creates is a `tempfile.TemporaryDirectory`
# (never a bare `mkdtemp`), kept alive by a reference in `_TEMP_DIRS` for
# the run's duration and `.cleanup()`-ed at process exit via `atexit` — a
# self-test run leaves nothing behind, deliberately, rather than trusting
# whatever ran it to sweep `/tmp` afterward.
_TEMP_DIRS: list[tempfile.TemporaryDirectory] = []


@atexit.register
def _cleanup_temp_dirs() -> None:
    for d in _TEMP_DIRS:
        d.cleanup()


def _fresh_root() -> Path:
    d = tempfile.TemporaryDirectory(prefix="lead-gate-selftest-")
    _TEMP_DIRS.append(d)
    return Path(d.name)


# esc-097 (HARD_BLOCK fix): a single, shared, empty directory — never a
# fixture's own `project_dir` — used as `_run()`'s default `cwd`. It carries
# no `.jammi/gate-state`, so a mutant that makes `repo_root()` prefer
# `Path.cwd()` over `CLAUDE_PROJECT_DIR` fails almost every fixture (state
# resolves to an empty decoy instead of the real fixture root), rather than
# passing all of them by accident.
_DECOY_CWD_TD = tempfile.TemporaryDirectory(prefix="lead-gate-decoy-cwd-")
_TEMP_DIRS.append(_DECOY_CWD_TD)
_DECOY_CWD = Path(_DECOY_CWD_TD.name)


# --------------------------------------------------------------------------
# esc-097 (R3, "probe the fix"): the harness's git dependency. A fixture that
# must reach an ACCEPTED relay under the esc-097 patch needs a real,
# CLAUDE_PROJECT_DIR-resolvable commit graph — `_temp_repo()` mints one (git
# init, identity via env, `commit.gpgsign=false`); `_write_block_row` then
# mints a REAL commit sha as `head_sha` instead of a `cafef00d` placeholder
# whenever `root` is such a repo. Every DENY fixture that never reaches the
# relay's git arm (R1/R2/"no relay artifact"/schema denials) is UNCHANGED —
# it still uses a plain, non-git `_fresh_root()` and the old placeholder sha
# (V5: "every existing DENY fixture must still deny with ITS reason").
# --------------------------------------------------------------------------

# Mirrors `_GIT_BUDGET_S` in the (proposed, patched) hook's `_run_git` —
# duplicated here rather than imported so this fixture harness states its
# own budget explicitly, independent of whatever the patched lib currently
# says.
_GIT_TIMEOUT_S_FIXTURE = 5.0

_GIT_FIXTURE_ENV_KEYS = {
    "GIT_AUTHOR_NAME": "lead-gate-fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "lead-gate-fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def _git_fixture_env() -> dict:
    env = dict(os.environ)
    env.update(_GIT_FIXTURE_ENV_KEYS)
    return env


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root)] + list(args), env=_git_fixture_env(),
                           capture_output=True, text=True, timeout=10)
    _assert(proc.returncode == 0, "git fixture setup",
            f"git {' '.join(args)} (in {root}) failed: {proc.stderr}")
    return proc.stdout.strip()


def _temp_repo(unit_branch: str) -> Path:
    """A fresh tempdir that is ALSO a real git repo (`git init`, identity via
    env, `commit.gpgsign=false`) on a branch named `unit_branch`, with one
    seed commit — this repo root doubles as `CLAUDE_PROJECT_DIR` for the
    fixture (`_run` already sets that env var to `root` on every hook call)."""
    root = _fresh_root()
    _git(root, "init", "-q")
    _git(root, "config", "commit.gpgsign", "false")
    # `.jammi/gate-state/*` (this SAME root's hook state) must NEVER be
    # swept up by a later `git add -A` in `_commit_fix` — a state file
    # accidentally committed on one branch and absent on another would be
    # DELETED by `git checkout` when switching branches (exactly the trap
    # G22's amend-sibling setup exercises: a checkout back to the unit
    # branch after committing on a throwaway branch). The real repo this
    # hook runs in already gitignores `.jammi/` for the same reason.
    (root / ".gitignore").write_text(".jammi/\n")
    (root / "SEED.md").write_text("seed\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed")
    _git(root, "checkout", "-q", "-b", unit_branch)
    return root


def _commit_fix(root: Path, *files: str) -> str:
    """Commits a change to each of `files` (creating/overwriting them with
    placeholder content unique to this call) on the CURRENT branch of the
    repo at `root` — the fixture's stand-in for "the fix" whose window R3
    requires a probe to name a member of. Returns the new commit's full
    sha."""
    marker = _git(root, "rev-parse", "HEAD") + str(len(files))
    for name in files:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"fix content for {name} ({marker})\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: " + ", ".join(files))
    return _git(root, "rev-parse", "HEAD")


def _commit_claim_fix(root: Path, path: str, lines: list[str]) -> str:
    """esc-lead-gate-R11: like `_commit_fix`, but writes EXACT `lines`
    content (never placeholder text) to `path`, on the current branch —
    fixtures need a real, known claim-shaped line at a known 1-indexed line
    number to exercise the hook's own `_parse_claim_sites` derivation."""
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", f"fix: {path}")
    return _git(root, "rev-parse", "HEAD")


def _claim_output_hash(rc: int, stdout: str) -> str:
    """MUST match `_run_claim_command`'s own hashing convention in the
    (proposed, patched) hook EXACTLY: `sha256(f"{rc}\\n{stdout}")` — stdout
    and exit code only, stderr excluded."""
    return hashlib.sha256(f"{rc}\n{stdout}".encode("utf-8")).hexdigest()


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
    constructing a relay artifact). `head_sha` is a REAL commit sha, minted
    fresh in `root`, whenever `root` is a git repo (`_temp_repo()`); a plain
    `_fresh_root()` root keeps the old `cafef00d` placeholder (fine for any
    fixture that denies before ever reaching the git arm)."""
    head_sha = (_commit_fix(root, f".block-{agent_type}-{verdict}.marker")
                if (root / ".git").exists()
                else "cafef00d1234567890abcdef1234567890abcdef")
    v: dict = {
        "kind": "verdict", "agent": agent_type, "diff_range": "base...head",
        "verdict": verdict, "uncertain": False, "unit_branch": unit_branch,
        "head_sha": head_sha,
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


# esc-lead-gate-R10: the fixture-only default `open_question` value every
# pre-R10 fixture inherits automatically (below) — never asserted on, only
# a valid non-empty placeholder so existing ALLOW fixtures stay ALLOW.
_DEFAULT_OPEN_QUESTION = "fixture-default.py:1 — attack: none captured (fixture placeholder)"


def _write_relay_exact(root: Path, row: dict, sites: dict[str, str] | None = None,
                        probe: list[str] | None = None, fix_head: str | None = None,
                        open_question: str | None = _DEFAULT_OPEN_QUESTION,
                        claims: dict[str, dict] | None = None,
                        override: dict | None = None) -> None:
    """esc-097: `fix_head` (the fix commit's full sha) is written into the
    relay artifact whenever the caller supplies one — the R3 arm's own
    field, always OMITTED unless a caller passes it (so every pre-esc-097
    fixture's relay shape is byte-identical to before).

    esc-lead-gate-R10: `open_question` defaults to a fixed, non-empty
    placeholder string — so every EXISTING call site (none of which passes
    this kwarg) keeps satisfying the new always-armed requirement without
    editing dozens of unrelated fixtures — exactly the same backward-compat
    shape `fix_head`'s own None-means-omitted default already established.
    A fixture that means to test the R10 arm itself passes
    `open_question=None` (omit) or an explicit string (present).

    esc-lead-gate-R11: `claims` needs NO such backward-compat default —
    the arm is armed by the DATA (a non-empty hook-derived `claim_sites`),
    and no EXISTING fixture's placeholder fix content (`"fix content for
    {name} ({marker})\\n"`, from `_commit_fix`) matches any claim phrase, so
    every pre-R11 fixture's `claim_sites` is empty and the arm never fires
    for them regardless of whether `claims` is present."""
    path = _relay_path_exact(root, row)
    artifact = {"unit_branch": row["unit_branch"], "agent_type": row["agent_type"], "block_ts": row["ts"]}
    if fix_head is not None:
        artifact["fix_head"] = fix_head
    if sites is not None:
        artifact["sites"] = sites
    if probe is not None:
        artifact["probe"] = probe
    if open_question is not None:
        artifact["open_question"] = open_question
    if claims is not None:
        artifact["claims"] = claims
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
    """esc-064: a full-coverage relay is no longer sufficient by itself —
    R2 (adjacent probing) is armed on this arm too, so the accepted relay
    now carries a probe array alongside full site coverage. esc-097: a real
    repo + `fix_head` (V5) so this stays ALLOWED once the R3 patch lands —
    `d.py` is both an adjacent probe site AND the fix's own changed file."""
    root = _temp_repo("feat/g6")
    row = _write_block_row(root, "feat/g6", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
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

    root2 = _temp_repo("feat/g8b")
    row2 = _write_block_row(root2, "feat/g8b", "a1", "adversarial-audit", None, ["foo.py:10"])
    fix_head2 = _commit_fix(root2, "baz.py")
    _write_relay_exact(root2, row2, probe=["bar.py:5", "baz.py:9"], fix_head=fix_head2)  # disjoint, >=2
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


def fixture_g13_cross_type_pass_irrelevant_only_same_type_relay_governs() -> None:
    """esc-097 (V10): there is no cross-type clearing arm at all — a
    fix-verifier PASS elsewhere is irrelevant to an adversarial-audit
    BLOCK's own repeat dispatch. G13a: no relay at all -> still denied
    (true regardless of the mechanism). G13b: a relay that is fully
    accepted for the adversarial-audit's OWN repeat dispatch (R1+R2, and —
    once the esc-097 patch lands — R3 via its own `fix_head`) allows,
    via the ONE gate's ordinary same-type rule; an unrelated fix-verifier
    PASS present in the state dir plays no role in that outcome. See G32
    for the round-2 reproducer this arm's deletion fixes: a relay that
    would NOT satisfy R3 (no `fix_head`) stays denied even WITH both a
    fix-verifier and an acceptance-verifier PASS on record."""
    root = _fresh_root()
    aa_row = _write_block_row(root, "feat/g13", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    # No relay yet: a fix-verifier PASS must NOT clear the audit BLOCK.
    _write_block_row(root, "feat/g13", "f1", "fix-verifier", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {aa_row['worktree']}"}}, root)
    _assert(p.returncode == 2, "G13a", f"fix-verifier PASS with NO relay must not clear, got {p.returncode}")

    root2 = _temp_repo("feat/g13b")
    aa_row2 = _write_block_row(root2, "feat/g13b", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    fix_head2 = _commit_fix(root2, "d.py")
    # esc-064: the accepted relay needs the always-armed probe here too.
    # esc-097: + a real fix_head (`d.py` doubles as the adjacent probe site
    # AND the fix's own changed file) -- this relay satisfies the SAME
    # gate's ordinary same-type rule on its own; the fix-verifier PASS
    # below is present but load-bearing for nothing.
    _write_relay_exact(root2, aa_row2, sites={"a.py:1": "fixed"}, probe=["c.py:9", "d.py:4"],
                        fix_head=fix_head2)
    _write_block_row(root2, "feat/g13b", "f1", "fix-verifier", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit at {aa_row2['worktree']}"}}, root2)
    _assert(p.returncode == 0, "G13b", f"a fully-accepted same-type relay must allow, got {p.returncode}: {p.stderr}")


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
    # esc-lead-gate-R7d: the vocabulary is PER AGENT TYPE, never pooled — a
    # raw value that is PASS-like for ONE card must NOT clear a DIFFERENT
    # card's row. All three reproduced true against the round-1/round-2
    # pooled `_PASS_LIKE` set (KILL verdicts tasks/abe09018c16727106.output
    # finding 7, tasks/a51ebcc1234dd552f.output premise "R6c").
    for raw, agent_type in (("verified", "adversarial-audit"), ("PROCEED", "adversarial-audit"),
                             ("verified", "oracle")):
        root = _fresh_root()
        v = {"kind": "verdict", "verdict": raw, "unit_branch": f"feat/neg-{raw}-{agent_type}", "findings": []}
        msg = "```json\n" + json.dumps(v) + "\n```"
        row = _stop_and_read(root, msg, agent_type=agent_type)
        _assert(row["verdict"] == "BLOCK", f"V4-6neg[{agent_type}/{raw}]",
                f"{raw!r} is not {agent_type}'s own PASS spelling and must NOT clear, got {row['verdict']!r}")


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


def fixture_g16_reactive_relay_rejected_when_enumeration_present() -> None:
    """esc-064 RED case: a BLOCK with a NON-EMPTY class_enumeration whose
    relay restates it as `sites` with NO `probe` array is NOT accepted.
    R2 (adjacent probing) is armed UNCONDITIONALLY — a reactive relay that
    only acknowledges the verifier's own enumeration is insufficient, and
    the deny reason must NAME the missing probe evidence so the lead's
    remedy is legible (not an uninterpretable deny that invites the rm
    escape hatch)."""
    root = _fresh_root()
    row = _write_block_row(root, "feat/g16", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _write_relay_exact(root, row, sites={"a.py:1": "noted", "b.py:2": "noted"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g16"}}, root)
    _assert(p.returncode == 2, "G16", f"reactive relay must deny(2), got {p.returncode}")
    _assert("probe" in p.stderr, "G16",
            f"the deny reason must name the missing probe evidence: {p.stderr!r}")


def fixture_g17_probed_relay_accepted_when_enumeration_present() -> None:
    """esc-064 GREEN counterpart (non-vacuity for G16/G18): the SAME
    full-coverage relay plus >=2 probe sites disjoint from the enumeration
    and every finding location IS accepted. esc-097: + a real repo +
    fix_head (V5) — `d.py` doubles as the adjacent probe AND the fix's own
    changed file."""
    root = _temp_repo("feat/g17")
    row = _write_block_row(root, "feat/g17", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "noted", "b.py:2": "noted"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g17"}}, root)
    _assert(p.returncode == 0, "G17", f"probed relay must accept, got {p.returncode}: {p.stderr}")


def fixture_g18_probe_boundary_cases() -> None:
    """esc-064 boundary matrix — each root isolates ONE axis and must deny.
    Roots (vi)/(vii)/(viii) are the mutation-adequacy additions from the
    proposal's pressure test: (vi) kills the exact-dedup/strip-collision
    mutant, (vii) kills the strip-probe-side-only mutant (a verifier-emitted
    PADDED finding location is reachable — locations are recorded verbatim),
    (viii) kills the plain-.strip() mutant (zero-width Cf characters are not
    whitespace)."""
    cases = [
        # (label, class_enum, finding_locs, probe)
        ("i-enumerated-collision", ["a.py:1", "b.py:2"], ["a.py:1"], ["b.py:2", "z.py:9"]),
        ("ii-count-1", ["a.py:1", "b.py:2"], ["a.py:1"], ["z.py:9"]),
        ("iii-empty-entries", ["a.py:1", "b.py:2"], ["a.py:1"], ["", "   ", "z.py:9"]),
        ("iv-duplicates", ["a.py:1", "b.py:2"], ["a.py:1"], ["z.py:9", "z.py:9"]),
        ("v-padded-collision", ["a.py:1", "b.py:2"], ["a.py:1"], ["a.py:1 ", "b.py:2"]),
        ("vi-strip-identical-pair", ["a.py:1", "b.py:2"], ["a.py:1"], [" x.py:1", "x.py:1 "]),
        ("vii-verifier-padded-finding", ["a.py:1"], ["foo.py:10 "], ["foo.py:10", "z.py:9"]),
        ("viii-zero-width-collision", ["a.py:1"], ["z.py:9"], ["z.py:9\u200b", "z.py:9\u200b\u200b"]),
    ]
    for label, enum, locs, probe in cases:
        root = _fresh_root()
        row = _write_block_row(root, "feat/g18", "a1", "adversarial-audit", enum, locs)
        _write_relay_exact(root, row, sites={s: "noted" for s in enum}, probe=probe)
        p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g18"}}, root)
        _assert(p.returncode == 2, "G18",
                f"case {label}: must deny (fewer than 2 ADJACENT sites), got {p.returncode}: {p.stderr}")


def fixture_g19_coverage_arm_selected_by_data_not_flag() -> None:
    """esc-064 mutation-adequacy fixture: R1 (coverage) is armed by the DATA
    (a non-empty class_enumeration), never by the recorded
    `enumeration_missing` flag — a row MISSING that key entirely (a
    legacy/hand-edited row shape) with a relay carrying `probe` but NO
    `sites` must still deny. A fix that merely bolts `and probe_ok` onto
    the old flag-selected arm leaves this fixture red; only removing the
    flag as a discriminator satisfies it.

    BUILDABILITY CAVEAT (stated, per the proposal's pressure test):
    `handle_stop` ALWAYS writes `enumeration_missing`, so `_write_block_row`
    cannot emit the key-absent row — this fixture post-edits the state
    JSONL to DELETE the key. Do NOT weaken it into a hook-emittable row."""
    root = _fresh_root()
    _write_block_row(root, "feat/g19", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    state = root / ".jammi" / "gate-state" / (_slug("feat/g19") + ".jsonl")
    rows = [json.loads(line) for line in state.read_text().splitlines() if line.strip()]
    for r in rows:
        r.pop("enumeration_missing", None)
    state.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    row = rows[-1]
    _write_relay_exact(root, row, probe=["z.py:9", "y.py:8"])  # probe, but NO sites
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g19"}}, root)
    _assert(p.returncode == 2, "G19",
            f"data-armed coverage must deny a probe-only relay, got {p.returncode}: {p.stderr}")


# ==========================================================================
# OQ1-OQ2 — esc-lead-gate-R10: alongside the >=2 examined-clean probe sites
# R2 already requires, ONE further relay entry must be an OPEN QUESTION — a
# site examined and explicitly NOT closed, naming the attack for the next
# round. Armed ALWAYS, conjunctive with R1-R3, never folded into `probe`'s
# own counting.
# ==========================================================================

def fixture_oq1_no_open_question_denies() -> None:
    """A relay with >=2 clean, disjoint probe sites AND a valid fix_head
    (satisfying R1/R2/R3 in full) but an explicitly ABSENT `open_question`
    must still be denied — the new field is armed UNCONDITIONALLY, never
    satisfied by `probe` alone."""
    root = _temp_repo("feat/oq1")
    row = _write_block_row(root, "feat/oq1", "a1", "adversarial-audit", None, ["foo.py:10"])
    fix_head = _commit_fix(root, "baz.py")
    _write_relay_exact(root, row, probe=["bar.py:5", "baz.py:9"], fix_head=fix_head,
                        open_question=None)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/oq1"}}, root)
    _assert(p.returncode == 2, "OQ1", f"a relay with no open_question must deny, got {p.returncode}: {p.stderr}")
    _assert("open_question" in p.stderr, "OQ1",
            f"the deny reason must name the missing open_question: {p.stderr!r}")


def fixture_oq2_open_question_present_allows() -> None:
    """The SAME relay as OQ1, plus a non-empty `open_question`, must ALLOW —
    proving the new field is satisfiable, not merely a permanent deny."""
    root = _temp_repo("feat/oq2")
    row = _write_block_row(root, "feat/oq2", "a1", "adversarial-audit", None, ["foo.py:10"])
    fix_head = _commit_fix(root, "baz.py")
    _write_relay_exact(
        root, row, probe=["bar.py:5", "baz.py:9"], fix_head=fix_head,
        open_question="qux.py:3 — examined, could not close: retry under a concurrent writer")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/oq2"}}, root)
    _assert(p.returncode == 0, "OQ2", f"a relay with an open_question present must allow, got {p.returncode}: {p.stderr}")


# ==========================================================================
# UC1-UC7 — esc-lead-gate-R11 ("untested claims carry a test"): the relay may
# not be accepted while the fix's OWN diff still carries a claim-shaped line
# (the HOOK's own derived enumeration, `_parse_claim_sites`) with no
# disposition in `claims`. Registered directly in FIXTURES below, the same
# way OQ1/OQ2 (esc-lead-gate-R10) already are — this file and the mechanism
# it exercises land in the SAME patch, so there is no unpatched state in
# which these fixture functions exist at all.
# ==========================================================================

_UC_ENUM = ["foo.py:10"]
_UC_FINDINGS = ["foo.py:10"]
_UC_PROBE = ["bar.py:2", "qux.py:1"]
_UC_SITES = {"foo.py:10": "checked and clean"}
_UC_CLAIM_LINES = [
    "import os",
    "# this cannot be driven from a test",
    "# no injection point here",
    "def f():",
    "    return os",
]


def _uc_setup(unit: str):
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", _UC_ENUM, _UC_FINDINGS)
    fix_head = _commit_claim_fix(root, "bar.py", _UC_CLAIM_LINES)
    return root, row, fix_head


def fixture_uc1_missing_claim_denied() -> None:
    """The fix adds a real claim-shaped line (`bar.py:2`, `# this cannot be
    driven from a test`) but the relay carries no `claims` object at all —
    denied naming the derived, uncovered claim-shaped line(s), even though
    R1/R2/R3/R10 are all otherwise satisfied."""
    root, row, fix_head = _uc_setup("feat/uc1")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc1"}}, root)
    _assert(p.returncode == 2, "UC1", f"a relay with no `claims` must deny, got {p.returncode}: {p.stderr}")
    _assert("claim" in p.stderr.lower(), "UC1",
            f"the deny reason must name the missing claims obligation: {p.stderr!r}")


def fixture_uc2_tested_claim_with_matching_hash_allows() -> None:
    """The SAME relay as UC1, with `bar.py:2` marked `tested` and a command
    whose RE-EXECUTED output hash matches the recorded one, must ALLOW —
    `bar.py:3` (`# no injection point here`) also needs a disposition since
    the hook enumerates BOTH claim-shaped lines the fix adds."""
    root, row, fix_head = _uc_setup("feat/uc2")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head, claims={
        "bar.py:2": {"status": "tested", "command": "printf hello",
                     "output_hash": _claim_output_hash(0, "hello")},
        "bar.py:3": {"status": "uncovered", "reason": "no fault-injecting backend in this fixture; "
                                                        "attack: drive it via the pattern in tests/it/probe.rs"},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc2"}}, root)
    _assert(p.returncode == 0, "UC2", f"a fully-dispositioned relay must allow, got {p.returncode}: {p.stderr}")


def fixture_uc3_hash_mismatch_denied() -> None:
    """The SAME `tested` claim as UC2, but the recorded `output_hash` does
    NOT match what re-executing `command` actually produces — denied,
    naming that the claim is not established."""
    root, row, fix_head = _uc_setup("feat/uc3")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head, claims={
        "bar.py:2": {"status": "tested", "command": "printf hello", "output_hash": "0" * 64},
        "bar.py:3": {"status": "uncovered", "reason": "no fault-injecting backend in this fixture"},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc3"}}, root)
    _assert(p.returncode == 2, "UC3", f"a non-reproducing hash must deny, got {p.returncode}: {p.stderr}")
    _assert("not established" in p.stderr or "does not reproduce" in p.stderr, "UC3",
            f"the deny reason must name the non-reproducing hash: {p.stderr!r}")


def fixture_uc4_write_verb_denied() -> None:
    """A `tested` claim whose command contains a denied write-verb program
    is denied WITHOUT ever executing it — never reaches the hash-compare
    step at all."""
    root, row, fix_head = _uc_setup("feat/uc4")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head, claims={
        "bar.py:2": {"status": "tested", "command": "rm -rf /tmp/should-never-run-uc4",
                     "output_hash": "a" * 64},
        "bar.py:3": {"status": "uncovered", "reason": "no fault-injecting backend in this fixture"},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc4"}}, root)
    _assert(p.returncode == 2, "UC4", f"a write-verb command must deny, got {p.returncode}: {p.stderr}")
    _assert("denied program" in p.stderr or "write-verb" in p.stderr, "UC4",
            f"the deny reason must name the write-verb denylist: {p.stderr!r}")
    _assert(not Path("/tmp/should-never-run-uc4").exists(), "UC4",
            "the denied command must never have been executed")


def fixture_uc5_uncovered_with_reason_allows() -> None:
    """Both claim-shaped lines marked `uncovered`, each with its OWN
    distinct, non-empty reason — must ALLOW; an honest disclosure costs
    nothing."""
    root, row, fix_head = _uc_setup("feat/uc5")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head, claims={
        "bar.py:2": {"status": "uncovered", "reason": "no fault-injecting backend in this fixture; "
                                                        "attack: drive it via tests/it/probe.rs's own pattern"},
        "bar.py:3": {"status": "uncovered", "reason": "the injection surface here is a private fn with "
                                                        "no test harness yet; attack: add one exercising it directly"},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc5"}}, root)
    _assert(p.returncode == 0, "UC5", f"two distinct honest uncovered reasons must allow, got {p.returncode}: {p.stderr}")


def fixture_uc6_duplicate_uncovered_reason_denied() -> None:
    """Both claim-shaped lines marked `uncovered` with the IDENTICAL
    (normalized) reason — a templated, copy-pasted disposition, denied as
    the anti-vacuity check's own target."""
    root, row, fix_head = _uc_setup("feat/uc6")
    same_reason = "not established; attack: write a probe"
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head, claims={
        "bar.py:2": {"status": "uncovered", "reason": same_reason},
        "bar.py:3": {"status": "uncovered", "reason": "  " + same_reason + "  "},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc6"}}, root)
    _assert(p.returncode == 2, "UC6", f"a templated duplicate reason must deny, got {p.returncode}: {p.stderr}")
    _assert("IDENTICAL" in p.stderr, "UC6",
            f"the deny reason must name the duplicate uncovered reason: {p.stderr!r}")


def fixture_uc7_no_claim_shaped_line_is_a_noop() -> None:
    """A fix that adds NO claim-shaped line (ordinary `_commit_fix`
    placeholder content) carries no R11 obligation at all — the relay needs
    no `claims` object, the same "armed by the DATA" posture R1 already
    takes toward `class_enumeration`."""
    root = _temp_repo("feat/uc7")
    row = _write_block_row(root, "feat/uc7", "a1", "adversarial-audit", _UC_ENUM, _UC_FINDINGS)
    fix_head = _commit_fix(root, "bar.py")
    _write_relay_exact(root, row, sites=_UC_SITES, probe=_UC_PROBE, fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/uc7"}}, root)
    _assert(p.returncode == 0, "UC7", f"a fix with no claim-shaped line must allow with no `claims`, "
                                       f"got {p.returncode}: {p.stderr}")


# ==========================================================================
# G20-G40 — esc-097 (R3, "probe the fix"; G32-G35 are the round-2, G36-G38
# the round-3, and G39-G40 the round-4 pressure-test reproducers). NOT
# added to FIXTURES: the hook patch these
# exercise is a PROPOSAL (`.claude/hooks/**` is agent-write-denied), so it
# is not applied in THIS tree. `_g20_28_arm()` (called from `self_test()`,
# mirroring how N7 already runs outside the FIXTURES loop) detects whether
# the lib on disk carries the R3 patch (`hasattr(mod, "RELAY_R3")`) and,
# when absent, reports the WHOLE arm SKIPPED and returns success for it
# ONLY — every fixture function below still runs unmodified, unchanged, the
# moment a human applies the proposal's patch files (verified in a
# THROWAWAY, session-local `cp -R` copy of the tree — never a git worktree
# of this repo, deleted after use — never in this checkout).
# ==========================================================================

def fixture_g20_no_fix_head_denies() -> None:
    root = _temp_repo("feat/g20")
    row = _write_block_row(root, "feat/g20", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"])  # no fix_head at all
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g20"}}, root)
    _assert(p.returncode == 2, "G20", f"a relay with no fix_head must deny, got {p.returncode}: {p.stderr}")
    _assert("fix_head" in p.stderr, "G20", f"deny reason must name fix_head: {p.stderr!r}")


def fixture_g21_fix_head_equals_block_sha_denies() -> None:
    root = _temp_repo("feat/g21")
    row = _write_block_row(root, "feat/g21", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=row["head_sha"])
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g21"}}, root)
    _assert(p.returncode == 2, "G21", f"fix_head == block_sha must deny (re-roll), got {p.returncode}: {p.stderr}")
    _assert("re-roll" in p.stderr, "G21", f"deny reason must name the re-roll, got: {p.stderr!r}")


def fixture_g22_relay_unit_branch_name_binds_reachability() -> None:
    """esc-097 V16+V18: reachability binds the unit's NAME git-free FIRST —
    the relay's own `unit_branch` must `slugify()` to EXACTLY this BLOCK's
    own `unit_slug` (the identity the dispatch already resolved), never an
    arbitrary branch name alone. (a) a relay naming the unit's OWN branch
    (whose tip its OWN `fix_head` is reachable from — see G25/G36/G37 for
    the POSITION half V18 adds on top of this NAME check) is ALLOWED; (b)
    on the SAME `fix_head`, a relay naming a DIFFERENT branch is DENIED by
    this NAME check alone, before the POSITION check is ever reached —
    reason says the relay does not name this BLOCK's own unit."""
    root = _temp_repo("feat/g22")
    row = _write_block_row(root, "feat/g22", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")

    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head,
                        override={"unit_branch": "feat/g22"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g22"}}, root)
    _assert(p.returncode == 0, "G22a",
            f"a relay naming the unit's own branch must allow, got {p.returncode}: {p.stderr}")

    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head,
                        override={"unit_branch": "feat/g22-unrelated"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g22"}}, root)
    _assert(p.returncode == 2, "G22b",
            f"a relay naming a DIFFERENT unit must deny, got {p.returncode}: {p.stderr}")
    _assert("does not name this BLOCK" in p.stderr, "G22b",
            f"deny reason must say the relay does not name this BLOCK's own unit: {p.stderr!r}")


def fixture_g23_no_probe_names_fix_changed_denies() -> None:
    root = _temp_repo("feat/g23")
    row = _write_block_row(root, "feat/g23", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "e.py:4"], fix_head=fix_head)  # neither is d.py
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g23"}}, root)
    _assert(p.returncode == 2, "G23", f"a probe naming no fix-changed file must deny, got {p.returncode}: {p.stderr}")
    _assert("probe the fix" in p.stderr, "G23", f"deny reason must redirect to the fix: {p.stderr!r}")


def fixture_g24_probe_names_fix_changed_finding_file_allows() -> None:
    """D3: a probe naming a fix-changed file that is ALSO a finding location
    DOES satisfy R3 — probing the fix's own surface is the point. R2's own
    >=2-distinct-non-reactive requirement stays conjunctive and is satisfied
    by the OTHER two probe entries."""
    root = _temp_repo("feat/g24")
    row = _write_block_row(root, "feat/g24", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "a.py")  # the fix touches the SAME file a finding named
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["a.py:1", "c.py:9", "d.py:4"], fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g24"}}, root)
    _assert(p.returncode == 0, "G24",
            f"a probe naming a fix-changed FINDING file must satisfy R3, got {p.returncode}: {p.stderr}")


def fixture_g25_amend_sibling_fix_allows_orphan_denies() -> None:
    """esc-097 V18 (round-3 closure): `fix_head` must be reachable from the
    unit branch's own tip (`git merge-base --is-ancestor fix_head
    <resolved tip>`), not merely resolve to SOME commit object.

    (a) sibling fix ON the unit branch: an `--amend`-style reset+recommit
    makes the new commit the CURRENT tip of `feat/g25` — `block_sha` is NOT
    an ancestor of `fix_head` (both descend from a common parent), but
    `fix_head` IS the branch's own tip, so this still ALLOWS.

    (b) orphaned pre-amend sha: a STALE relay names an EARLIER, now
    amended-away commit that is neither `block_sha` (which would instead
    deny as a re-roll) nor the current tip — it still resolves via `git
    cat-file -e` (the object is not yet gc'd) but is NOT an ancestor of the
    unit branch's tip — DENY, naming the ancestry failure and the remedy
    (name the amended sha). This is the round-3 adversarial reproducer:
    RED against the 7633b2d6 patch (which never checked `fix_head`'s
    position at all, only the relay's `unit_branch` NAME), GREEN once
    V18's ancestry check lands."""
    root = _temp_repo("feat/g25")
    row = _write_block_row(root, "feat/g25", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    block_sha = row["head_sha"]
    _git(root, "reset", "-q", "--hard", "HEAD~1")  # back to block_sha's own parent
    orphan_sha = _commit_fix(root, "e.py")  # an in-between amend, later abandoned
    _git(root, "reset", "-q", "--hard", "HEAD~1")  # abandon it too, back to the same parent
    fix_head = _commit_fix(root, "d.py")  # the FINAL amended commit -- the real tip
    _assert(len({block_sha, orphan_sha, fix_head}) == 3, "G25 setup", "must be three distinct shas")
    p_ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", block_sha, fix_head])
    _assert(p_ancestor.returncode != 0, "G25 setup",
            "block_sha must NOT be an ancestor of fix_head (the amend-sibling premise)")
    p_orphan = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", orphan_sha, fix_head])
    _assert(p_orphan.returncode != 0, "G25 setup",
            "the in-between (abandoned) amend must NOT be an ancestor of the final tip either")

    # (a) sibling fix ON the unit branch's own current tip -> ALLOW.
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g25"}}, root)
    _assert(p.returncode == 0, "G25a",
            f"an amend-sibling fix ON the unit branch's own tip must allow, got {p.returncode}: {p.stderr}")

    # (b) orphaned pre-amend sha -- resolves (not yet gc'd), but is NOT on
    # the unit branch's tip -> DENY.
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "e.py:4"], fix_head=orphan_sha)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g25"}}, root)
    _assert(p.returncode == 2, "G25b",
            f"an orphaned pre-amend fix_head (resolves, not on the tip) must deny, got {p.returncode}: {p.stderr}")
    _assert("is not on" in p.stderr, "G25b",
            f"deny reason must name the ancestry failure: {p.stderr!r}")


def fixture_g36_orphaned_sha_reachable_from_no_ref_denies() -> None:
    """esc-097 V18, round-3 adversarial reproducer: a `fix_head` that
    resolves as a real commit object (`git cat-file -e` succeeds — not yet
    gc'd) but is reachable from NO ref at all (a descendant abandoned by
    resetting the branch back). RED against the 7633b2d6 patch (V16-only:
    slug equality plus an independent `cat-file -e` resolution for each
    sha, no ancestry check at all — this shape ALLOWED); GREEN once V18's
    `git merge-base --is-ancestor` check lands."""
    root = _temp_repo("feat/g36")
    row = _write_block_row(root, "feat/g36", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    block_sha = row["head_sha"]
    # A commit that's a DESCENDANT of block_sha, then abandoned by resetting
    # feat/g36's own tip back to block_sha -- dangling, reachable only by
    # its raw sha, from no ref.
    orphan_sha = _commit_fix(root, "d.py")
    _git(root, "reset", "-q", "--hard", block_sha)
    _assert(orphan_sha != block_sha, "G36 setup", "must be a real, distinct sha")
    p_cat = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{orphan_sha}^{{commit}}"])
    _assert(p_cat.returncode == 0, "G36 setup", "the orphan sha must still resolve as an object")
    p_ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", orphan_sha, block_sha])
    _assert(p_ancestor.returncode != 0, "G36 setup",
            "the orphan sha must NOT be an ancestor of feat/g36's own (reset-back) tip")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=orphan_sha)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g36"}}, root)
    _assert(p.returncode == 2, "G36",
            f"an orphaned fix_head reachable from no ref must deny, got {p.returncode}: {p.stderr}")
    _assert("is not on" in p.stderr, "G36", f"deny reason must name the ancestry failure: {p.stderr!r}")


def fixture_g37_sha_on_unrelated_branch_denies() -> None:
    """esc-097 V18, round-3 adversarial reproducer: a `fix_head` that is a
    REAL, ref-reachable commit — just not on THIS unit's own branch. RED
    against the 7633b2d6 patch (no ancestry check — ANY resolvable sha
    ALLOWED regardless of which branch it actually lives on); GREEN once
    V18's `git merge-base --is-ancestor` check lands."""
    root = _temp_repo("feat/g37")
    row = _write_block_row(root, "feat/g37", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    _git(root, "checkout", "-q", "-b", "feat/g37-other")
    other_fix = _commit_fix(root, "d.py")
    _git(root, "checkout", "-q", "feat/g37")
    p_ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", other_fix, "feat/g37"])
    _assert(p_ancestor.returncode != 0, "G37 setup",
            "the other branch's fix must NOT be an ancestor of feat/g37's own tip")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=other_fix)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g37"}}, root)
    _assert(p.returncode == 2, "G37",
            f"a fix_head on an UNRELATED branch must deny, got {p.returncode}: {p.stderr}")
    _assert("is not on" in p.stderr, "G37", f"deny reason must name the ancestry failure: {p.stderr!r}")


def fixture_g26_claude_project_dir_unset_denies() -> None:
    root = _temp_repo("feat/g26")
    row = _write_block_row(root, "feat/g26", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g26"}}, root,
        env_overrides={"CLAUDE_PROJECT_DIR": None}, cwd=root)
    _assert(p.returncode == 2, "G26",
            f"an unset CLAUDE_PROJECT_DIR must deny in the relay arm, got {p.returncode}: {p.stderr}")
    _assert("CLAUDE_PROJECT_DIR" in p.stderr, "G26", f"deny reason must name CLAUDE_PROJECT_DIR: {p.stderr!r}")


def fixture_g27_git_timeout_denies() -> None:
    """V13: the shim spawns a PORTABLE escaped grandchild (`python3 -c
    "import os,time;os.setsid();time.sleep(30)" &`, backgrounded — its own
    `os.setsid()` moves it into a BRAND NEW session/process group, escaping
    whatever group the shim itself was started in) before `exec`ing a
    direct child that returns quickly on its own (`sleep 7`). Because the
    grandchild's own `setsid()` moved it out of the original group,
    `os.killpg(proc.pid, ...)` targeting that ORIGINAL group can never
    reach it — measured (see the proposal doc's "Bugs found and fixed" #2)
    against the round-1 PATCH AS COMMITTED (`Popen(..., stdout=PIPE,
    stderr=PIPE).communicate(timeout=5)`, `killpg` on timeout, THEN a
    second, UN-timed `proc.communicate()` "to reap"): ~30.03s, because that
    second call blocks until every pipe writer — including the escaped
    grandchild — closes its end. The FIXED `_run_git` (stdout/stderr to
    `tempfile.TemporaryFile()`s, never a pipe; bounded by `Popen.wait()`,
    never `.communicate()`) sidesteps the question of whether the
    grandchild is reachable at all — DENIES within T+2s regardless."""
    root = _temp_repo("feat/g27")
    row = _write_block_row(root, "feat/g27", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
    shim_bin = root / "git-shim-bin"
    shim_bin.mkdir()
    fake_git = shim_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'python3 -c "import os,time;os.setsid();time.sleep(30)" &\n'
        "exec sleep 7\n"
    )
    fake_git.chmod(0o755)
    real_path = os.environ.get("PATH", "")
    start = time.monotonic()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g27"}}, root,
        env_overrides={"PATH": f"{shim_bin}:{real_path}"})
    elapsed = time.monotonic() - start
    _assert(p.returncode == 2, "G27", f"a git call that times out must deny, got {p.returncode}: {p.stderr}")
    _assert("timed out" in p.stderr, "G27", f"deny reason must name the timeout: {p.stderr!r}")
    _assert(elapsed < _GIT_TIMEOUT_S_FIXTURE + 2.0, "G27",
            f"an escaped-grandchild git must still deny within T+2s, took {elapsed:.2f}s")


# --------------------------------------------------------------------------
# G28 — the REAL feat/deploy-shapes-E1-arm64-ci-base corpus (esc-097's own
# motivating incident). `_E1_ROUND_FIX_CHANGED` is the REAL, computed
# `git diff --name-only <round-N head> <round-(N+1) head>` file set for each
# of the five recorded adversarial-audit BLOCK rounds — read-only, against
# the actual project checkout's object store (shas 3f5b5bf0/cf0d37e7/
# cc9a59a3/8c7dec92/38bf6b44/d73ab690), pinned here as DATA (this fixture
# must be hermetic — it never touches the real repo). `_E1_ROUND_PROBE` is
# the REAL `probe` array from each of the five recorded relay artifacts
# (`.jammi/gate-state/feat_deploy-shapes-E1-arm64-ci-base.relay.adversarial-
# audit.*.json`), path tokens verbatim, with each entry's own descriptive
# tail kept (outcome-neutral — see `_probe_path`) (the round-5 relay already carries its own
# real `fix_head`; rounds 1-4 predate that field, so this fixture supplies
# the actual NEXT round's own recorded head_sha as fix_head, as the lead
# would have written it). The `class_enumeration` used to build each BLOCK
# row is a REAL 4-element TRUNCATION of that round's own recorded array (the
# real rows carry anywhere from 4 to 20 entries) — outcome-neutral: R1/R2
# only need >=1 matching entry and >=2 non-reactive probe sites respectively,
# neither of which this fixture's outcome depends on the full count for; the
# `finding_locations` used is the truncated array's own first element, also
# outcome-neutral for the same reason. The outcomes below are
# RECORDED BY RUNNING THE PATCHED CODE ONCE AND REVIEWING (V3/V5) — not
# asserted in advance: round 3 is the ONLY one of the five whose real probe
# names zero real fix-changed files; the other four ALSO name a real
# fix-changed file, mostly by the coincidence that this CI-infra unit
# repeatedly touches the same small hot-file set across rounds. This is
# reported HONESTLY in the proposal doc as R3's real, partial efficacy
# against its own motivating corpus, not oversold as "catches every round"."""
_E1_ROUND_FIX_CHANGED: list[list[str]] = [
    # round1 (block 3f5b5bf0) -> round2 head (cf0d37e7)
    [".cargo/config.toml", ".docker/ci-cuda.Dockerfile", ".github/workflows/_ci-base-image.yml",
     ".github/workflows/_e1-arm-probe.yml", ".github/workflows/ci.yml", ".github/workflows/image-cuda.yml",
     "CHANGELOG.md", "ci/scripts/check_merged_index_platforms.sh", "ci/scripts/test_check_gpu_prove_once.py"],
    # round2 (block cf0d37e7) -> round3 head (cc9a59a3)
    [".cargo/config.toml", ".github/actions/setup-rust-ci/action.yml", ".github/workflows/_ci-base-image.yml",
     ".github/workflows/image-cuda.yml", ".github/workflows/image.yml", "CHANGELOG.md",
     "docs/maintainer/MAINTAINER-GUIDE.md"],
    # round3 (block cc9a59a3) -> round4 head (8c7dec92)
    [".cargo/config.toml", ".github/actions/setup-rust-ci/action.yml", ".github/workflows/_ci-base-image.yml",
     "ci/scripts/rust_target_features.sh"],
    # round4 (block 8c7dec92) -> round5 head (38bf6b44)
    [".github/actions/setup-rust-ci/action.yml", ".github/workflows/_ci-base-image.yml",
     ".github/workflows/ci.yml", "ci/scripts/rust_target_features.sh"],
    # round5 (block 38bf6b44) -> round6 head (d73ab690) -- the real relay's OWN fix_head
    [".cargo/config.toml", ".github/actions/setup-rust-ci/action.yml", ".github/workflows/_ci-base-image.yml",
     ".github/workflows/_pypi-server.yml", ".github/workflows/ci.yml", ".github/workflows/cookbook-book.yml",
     ".github/workflows/cookbook-render.yml", ".github/workflows/dep-dag.yml", ".github/workflows/docs.yml",
     ".github/workflows/pypi.yml", ".github/workflows/release-binaries.yml", "CHANGELOG.md",
     "ci/scripts/rust_target_features.sh", "docs/maintainer/MAINTAINER-GUIDE.md"],
]

# The REAL `probe` arrays: path tokens verbatim from the five relay
# artifacts, each still carrying its own descriptive tail — outcome-neutral
# (R3 resolves each entry through `_probe_path`, which reads only the
# leading path token and never the prose after it).
_E1_ROUND_PROBE: list[list[str]] = [
    [".github/actions/docker-publish/action.yml:106 -- the hard `platforms: linux/amd64` literal in the sibling action: examined, deliberately untouched by E-1",
     ".github/workflows/image.yml:12-15 -- the paths: filter gains .github/workflows/_ci-base-image.yml so a reusable-only change rebuilds the base: examined at head (clean)",
     ".cargo/config.toml:9 vs :23 -- the x86_64 stanza keeps mold only; the aarch64 stanza carries mold + +fp16: examined; no other target stanza sets rustflags (clean)",
     ".github/actions/setup-rust-ci/action.yml:44-45 -- the composite's rustflags input REPLACES config rustflags when set: examined",
     "ci/scripts/check_merged_index_platforms.sh --self-test -- four fixtures wired into ci.yml:1467 guard matrix: examined and run, exit 0 (clean)"],
    [".github/workflows/release-binaries.yml:104-106 -- the deferred linux-aarch64 comment still stands (E-2 adds the row): examined, clean",
     ".devcontainer/Dockerfile:1-52 -- FROM …-ci:latest with no --platform pin and no arch-hardcoded download: examined, clean",
     "ci/scripts/pod_seed_target.sh:956 -- a RUSTFLAGS handler in the GPU pod tooling: x86_64 CUDA pods only, never aarch64 (examined, clean)",
     ".github/workflows/image.yml:12-16 and image-cuda.yml:13-18 -- paths filters now include ci/scripts/check_merged_index_platforms.sh (examined at head, clean)"],
    ["ci/scripts/rust_pin.sh -- the precedent for reading a pinned value out of a file in CI (rust-toolchain.toml): unwired from the guard matrix by convention (examined, clean)",
     ".github/workflows/dep-dag.yml:64 -- the only step-level RUSTFLAGS setter; amd64-only job: out of the composite's reach (examined, clean)",
     ".devcontainer/Dockerfile:1-52 -- no arch-hardcoded download; native arm64 build proved once (examined, clean)",
     "ci/scripts/check_merged_index_platforms.sh:135 -- ${1:?} guards the empty platform set: self-test 4/4 (examined and run, clean)"],
    [".github/workflows/ci.yml guard matrix -- rust_target_features.sh --self-test wired beside check_merged_index_platforms.sh --self-test (examined and run, clean)",
     "ci/scripts/rust_pin.sh:12 -- the fail-closed precedent the extractor now actually matches (examined, clean)",
     ".github/workflows/pypi.yml:39-44 -- job-level defaults.run.working-directory: packaging/native beside a setup-rust-ci call (examined, clean)",
     "ci/scripts/gpu-dev.sh:675,1099 / pod_seed_target.sh:239 -- pre-existing sites outside this unit's obligation (examined, left)"],
    [".github/actions/setup-rust-ci/action.yml:76-100 (FIX'S OWN NEW SURFACE) -- deny-warnings/target inputs, host triple from rustc -vV, fail-closed on empty/malformed triple (commit 13) (examined, clean)",
     ".github/workflows/ci.yml arm64-floor-oracle (FIX'S OWN NEW SURFACE) -- three unconditional asserts, dtolnay toolchain pinned via rust_pin.sh (examined, clean)",
     ".github/workflows/dep-dag.yml:64 -- migrated from a bare RUSTFLAGS step env to the per-target variable (examined, clean)",
     "the nine setup-rust-ci call sites -- every `rustflags:` input removed (examined, clean)",
     ".cargo/config.toml aarch64 stanza + CHANGELOG + MAINTAINER-GUIDE -- 'silent lower floor' language removed (examined, clean)"],
]

_E1_ROUND_CLASS_ENUM: list[list[str]] = [
    [".github/workflows/_ci-base-image.yml:266", ".github/workflows/_ci-base-image.yml:260",
     ".github/actions/docker-publish/action.yml:65", "CHANGELOG.md:454"],
    [".cargo/config.toml:23", ".github/actions/setup-rust-ci/action.yml:26",
     ".github/workflows/release-binaries.yml:97", "docs/maintainer/MAINTAINER-GUIDE.md:3753"],
    [".github/actions/setup-rust-ci/action.yml:72", ".cargo/config.toml:26",
     ".github/workflows/_ci-base-image.yml:294", ".github/workflows/dep-dag.yml:64"],
    [".github/actions/setup-rust-ci/action.yml:84", "ci/scripts/rust_target_features.sh:53",
     ".github/workflows/_ci-base-image.yml:252", ".cargo/config.toml:29"],
    [".github/actions/setup-rust-ci/action.yml:77", ".github/workflows/_ci-base-image.yml:316",
     ".github/workflows/dep-dag.yml:64", "ci/scripts/rust_target_features.sh:133"],
]

# (block short sha, "next round" / real fix_head short sha) per round, for
# assertion messages only.
_E1_ROUND_SHAS = [
    ("3f5b5bf0", "cf0d37e7"), ("cf0d37e7", "cc9a59a3"), ("cc9a59a3", "8c7dec92"),
    ("8c7dec92", "38bf6b44"), ("38bf6b44", "d73ab690"),
]


def fixture_g28_real_e1_corpus() -> None:
    root = _temp_repo("feat/deploy-shapes-E1-arm64-ci-base")
    outcomes = []
    for i in range(5):
        class_enum = _E1_ROUND_CLASS_ENUM[i]
        row = _write_block_row(root, "feat/deploy-shapes-E1-arm64-ci-base", "a1",
                                "adversarial-audit", class_enum, class_enum[:1])
        fix_head = _commit_fix(root, *_E1_ROUND_FIX_CHANGED[i])
        sites = {s: "fixed/examined at head" for s in class_enum}
        _write_relay_exact(root, row, sites=sites, probe=_E1_ROUND_PROBE[i], fix_head=fix_head)
        p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
            "subagent_type": "adversarial-audit",
            "prompt": "re-audit unit: feat/deploy-shapes-E1-arm64-ci-base"}}, root)
        block_short, fix_short = _E1_ROUND_SHAS[i]
        outcomes.append((i + 1, block_short, fix_short, p.returncode, p.stderr.strip()))

    # The REAL, computed result (run once, reviewed — V3/V5): R3 denies
    # ONLY round 3 (cc9a59a3 -> 8c7dec92) — its real probe (rust_pin.sh,
    # dep-dag.yml, .devcontainer/Dockerfile, check_merged_index_platforms.sh)
    # names none of the four real fix-changed files. Every other round's
    # real probe ALSO happens to name a real fix-changed file (a hot-file
    # coincidence in this CI-infra unit, not R3 catching adjacent-only
    # probing generally) -- stated honestly, not oversold.
    expected_allow = {1, 2, 4, 5}
    expected_deny_round = 3
    for round_no, block_short, fix_short, rc, reason in outcomes:
        if round_no == expected_deny_round:
            _assert(rc == 2, "G28",
                    f"round {round_no} (block {block_short}..fix {fix_short}): expected R3 to DENY "
                    f"(its real probe names no real fix-changed file), got {rc}: {reason!r}")
            _assert("probe the fix" in reason, "G28",
                    f"round {round_no} deny reason must redirect to the fix: {reason!r}")
        elif round_no in expected_allow:
            _assert(rc == 0, "G28",
                    f"round {round_no} (block {block_short}..fix {fix_short}): expected ALLOW (its real "
                    f"probe also names a real fix-changed file), got {rc}: {reason!r}")


def fixture_g29_first_dispatch_stays_git_free_with_hung_shim() -> None:
    """esc-097 (adversarial B2 fix): R3's git arm is bound to REPEAT
    dispatches only — never a first dispatch, and never through more units
    than the ONE the dispatch itself targets. Three OTHER units in the SAME
    state dir each carry an open BLOCK with an accepted (fully-relayed,
    fix_head-carrying) relay artifact — a busy, realistic gate-state — but
    the actual dispatch under test names NONE of them, so it is a genuine
    first dispatch of a FOURTH unit. A `git` on PATH that would hang for 30s
    if ever invoked must never be reached: the call returns in well under
    1s, and the shim's own invocation log (it appends before hanging) stays
    empty, proving zero git processes were spawned."""
    root = _temp_repo("feat/g29-decoy1")
    for i, branch in enumerate(["feat/g29-decoy1", "feat/g29-decoy2", "feat/g29-decoy3"]):
        if i:
            _git(root, "checkout", "-q", "-b", branch, "feat/g29-decoy1")
        row = _write_block_row(root, branch, f"a{i}", "adversarial-audit",
                                ["a.py:1", "b.py:2"], ["a.py:1"])
        fix_head = _commit_fix(root, f"d{i}.py")
        _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                            probe=["c.py:9", f"d{i}.py"], fix_head=fix_head)
    _git(root, "checkout", "-q", "feat/g29-decoy1")

    shim_bin = root / "git-shim-bin"
    shim_bin.mkdir()
    fake_git = shim_bin / "git"
    log_path = root / "git-invocations.log"
    fake_git.write_text(
        "#!/bin/sh\n"
        f"echo invoked >> {log_path}\n"
        "sleep 30\n"
    )
    fake_git.chmod(0o755)
    real_path = os.environ.get("PATH", "")
    start = time.monotonic()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": "unit: feat/g29-fresh\nFIRST audit of a brand-new unit, naming none of the decoys",
    }}, root, env_overrides={"PATH": f"{shim_bin}:{real_path}"})
    elapsed = time.monotonic() - start
    _assert(p.returncode == 0, "G29", f"a genuine first dispatch must allow, got {p.returncode}: {p.stderr}")
    _assert(elapsed < 1.0, "G29", f"a first dispatch must return in well under 1s, took {elapsed:.2f}s")
    _assert(not log_path.exists(), "G29",
            f"a first dispatch must spawn ZERO git processes, but the shim log exists: "
            f"{log_path.read_text() if log_path.exists() else ''!r}")


def fixture_g30_env_precedence_over_cwd() -> None:
    """esc-097 (HARD_BLOCK fix): `CLAUDE_PROJECT_DIR` governs state-dir
    resolution even when the subprocess's own `cwd` points at an entirely
    DIFFERENT directory that itself looks like a plausible root (it has its
    own `.jammi/gate-state`, just with no open BLOCK in it). A mutant that
    reduced `repo_root()` to `Path.cwd()` would resolve state against the
    WRONG (decoy) root here and find nothing to gate — this fixture ALLOWS
    only under that mutant; the real code DENIES because it reads the real
    root (named by the env var) and finds the open BLOCK there."""
    root = _temp_repo("feat/g30")
    row = _write_block_row(root, "feat/g30", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    decoy_cwd = _fresh_root()
    (decoy_cwd / ".jammi" / "gate-state").mkdir(parents=True)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit",
        "prompt": f"re-audit at {row['worktree']}",
    }}, root, cwd=decoy_cwd)
    _assert(p.returncode == 2, "G30",
            f"CLAUDE_PROJECT_DIR must govern over a decoy cwd, got {p.returncode}: {p.stderr}")


def fixture_g31_unbound_row_never_satisfiable_by_any_relay_branch() -> None:
    """esc-097 V16 (the B3 widening reverted): a BLOCK row whose OWN
    `unit_branch` is empty (the `UNBOUND` fallback bucket — no binding
    resolved at verdict-write time, so it is filed under `UNBOUND.jsonl`)
    can NEVER be satisfied by R3, regardless of what branch the relay
    names — no real branch name `slugify()`s to the literal string
    `UNBOUND`. The deny reason states the remedy: re-dispatch naming the
    unit so the verdict lands on the unit's own file, then hand-remove the
    stale row for this block from `UNBOUND.jsonl` (never `rm` the shared
    file — it holds every other unit's UNBOUND rows too)."""
    root = _temp_repo("feat/g31-real")
    row = _write_block_row(root, "", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head,
                        override={"unit_branch": "feat/g31-real"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit commit {row['head_sha']}"}}, root)
    _assert(p.returncode == 2, "G31",
            f"an UNBOUND row must never be satisfiable, got {p.returncode}: {p.stderr}")
    _assert("UNBOUND fallback bucket" in p.stderr, "G31",
            f"deny reason must name the UNBOUND remedy: {p.stderr!r}")


def fixture_g32_round2_no_cross_type_clearing() -> None:
    """V10, the round-2 reproducer: a relay WITHOUT `fix_head` (so it could
    never satisfy R3 on a direct dispatch either), plus BOTH a fix-verifier
    PASS and an acceptance-verifier PASS on record (chronologically after
    the BLOCK) — under the deleted cross-type clearing arm, this exact
    shape used to ALLOW, because that arm's own acceptance check
    (`check_fix=False`) never ran R3 at all. With the arm deleted, a repeat
    adversarial-audit dispatch naming this unit is STILL denied."""
    root = _temp_repo("feat/g32")
    aa_row = _write_block_row(root, "feat/g32", "a1", "adversarial-audit",
                               ["a.py:1", "b.py:2"], ["a.py:1"])
    _write_relay_exact(root, aa_row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"])  # no fix_head at all
    _write_block_row(root, "feat/g32", "f1", "fix-verifier", [], [], verdict="PASS")
    _write_block_row(root, "feat/g32", "v1", "acceptance-verifier", [], [], verdict="PASS")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g32"}}, root)
    _assert(p.returncode == 2, "G32",
            f"a relay with no fix_head must still deny despite two OTHER PASS rows, got {p.returncode}: {p.stderr}")


def _g33_g34_setup(order: str) -> tuple[Path, dict, dict, str]:
    root = _temp_repo("feat/g33-a")
    row_a = _write_block_row(root, "feat/g33-a", "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _git(root, "checkout", "-q", "-b", "feat/g33-b", "feat/g33-a")
    row_b = _write_block_row(root, "feat/g33-b", "a2", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _git(root, "checkout", "-q", "feat/g33-a")
    if order == "a-then-b":
        prompt = f"unit: none\nre-audit unit: feat/g33-a AND unit: feat/g33-b"
    else:
        prompt = f"unit: none\nre-audit unit: feat/g33-b AND unit: feat/g33-a"
    return root, row_a, row_b, prompt


def fixture_g33_two_units_denied_order_a() -> None:
    """V11: a prompt whole-token-naming MORE THAN ONE open BLOCK of the
    same type is denied outright, naming both — R3 is evaluated for
    exactly one unit per dispatch; the remedy is separate dispatches."""
    root, row_a, row_b, prompt = _g33_g34_setup("a-then-b")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": prompt}}, root)
    _assert(p.returncode == 2, "G33", f"naming two units must deny, got {p.returncode}: {p.stderr}")
    _assert("feat_g33-a" in p.stderr and "feat_g33-b" in p.stderr, "G33",
            f"deny reason must name BOTH units: {p.stderr!r}")
    _assert("one unit per dispatch" in p.stderr, "G33", f"reason must state the rule: {p.stderr!r}")


def fixture_g34_two_units_denied_order_b() -> None:
    """V11, the OTHER name order — the DENY (and both names appearing in
    the reason) must not depend on which unit's anchor the prompt mentions
    first."""
    root, row_a, row_b, prompt = _g33_g34_setup("b-then-a")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": prompt}}, root)
    _assert(p.returncode == 2, "G34", f"naming two units must deny, got {p.returncode}: {p.stderr}")
    _assert("feat_g33-a" in p.stderr and "feat_g33-b" in p.stderr, "G34",
            f"deny reason must name BOTH units regardless of order: {p.stderr!r}")


def fixture_g35_argv_boundary_head_sha_option_shaped() -> None:
    """V15: a BLOCK row's `head_sha` shaped like a git option
    (`--output=<path>`) must be rejected by the `re.fullmatch(r"[0-9a-f]
    {7,40}")` shape check BEFORE it is ever placed in a git argv (never
    relying on `--end-of-options` alone) — DENY, and the named output path
    is never written (proof that no git process ever acted on the value)."""
    root = _temp_repo("feat/g35")
    row = _write_block_row(root, "feat/g35", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    poc_path = root / "poc-output"
    malicious_sha = f"--output={poc_path}"
    state_file = root / ".jammi" / "gate-state" / "feat_g35.jsonl"
    rows = [json.loads(line) for line in state_file.read_text().splitlines() if line.strip()]
    rows[-1]["head_sha"] = malicious_sha
    state_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    fix_head = _commit_fix(root, "d.py")
    _write_relay_exact(root, rows[-1], sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g35"}}, root)
    _assert(p.returncode == 2, "G35",
            f"an option-shaped head_sha must deny, got {p.returncode}: {p.stderr}")
    _assert(not poc_path.exists(), "G35",
            f"the malicious head_sha's named output path must never be written: "
            f"exists={poc_path.exists()}")


def fixture_g38_unresolvable_fix_head_deny_reason_includes_git_stderr() -> None:
    """esc-097 V19: `_run_git` reads `err_f` AFTER `wait()` returns and
    appends its text to the deny reason — proven with a `fix_head` that is
    SHA-SHAPED (passes the `re.fullmatch` check) but names no real object
    at all. Round-4: `fix_head` is now resolved via `git rev-parse --verify`
    (not `cat-file -e`, so its OWN resolved value can be checked against the
    given hex — see G40), whose stderr for a wholly unresolvable hex is
    "fatal: Needed a single revision" (measured directly, `git 2.50.1`; NOT
    `cat-file -e`'s "Not a valid object name ..." — a different subcommand,
    a different message) — that exact text must appear in the hook's deny
    reason, not merely a bare "exited 128"."""
    root = _temp_repo("feat/g38")
    row = _write_block_row(root, "feat/g38", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fake_fix_head = "abc1234abc1234abc1234abc1234abc1234abcd"  # sha-shaped, no such object
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=fake_fix_head)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g38"}}, root)
    _assert(p.returncode == 2, "G38", f"an unresolvable fix_head must deny, got {p.returncode}: {p.stderr}")
    _assert("needed a single revision" in p.stderr.lower(), "G38",
            f"deny reason must include git's own stderr text, not just the exit code: {p.stderr!r}")


def fixture_g39_tag_shadowing_unit_branch_denies() -> None:
    """Round-4 adversarial reproducer: a TAG literally named EXACTLY like
    the relay's `unit_branch`, pointing at ANOTHER branch's commit that DOES
    contain `fix_head` as an ancestor, must not let a bare `<name>^{commit}`
    lookup (gitrevisions(7): `refs/tags/<name>` is disambiguated BEFORE
    `refs/heads/<name>`) resolve `unit_branch` to the TAG's target instead
    of the real branch's own tip, from which `fix_head` is NOT actually
    reachable. Measured directly (git 2.50.1): with both a branch and a tag
    named `feat/g39` in the same repo, `git rev-parse --verify
    feat/g39^{commit}` prints `warning: refname 'feat/g39' is ambiguous.` to
    stderr and resolves to the TAG's target anyway, exit 0 — silently a
    DIFFERENT commit than the real branch's tip. RED against 7ed0db7d's
    patch (bare resolution let the tag win and ALLOWED this exact shape);
    GREEN once `unit_branch` resolves under `refs/heads/` only."""
    root = _temp_repo("feat/g39")
    row = _write_block_row(root, "feat/g39", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    # feat/g39's own tip never advances past the BLOCK -- fix_head is NOT
    # reachable from it. The "fix" instead lands on an unrelated sibling
    # branch, forked from the same BLOCK commit.
    _git(root, "checkout", "-q", "-b", "feat/g39-other")
    other_fix = _commit_fix(root, "d.py")
    _git(root, "checkout", "-q", "feat/g39")
    p_ancestor = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                                  other_fix, "feat/g39"])
    _assert(p_ancestor.returncode != 0, "G39 setup",
            "the other branch's fix must NOT be an ancestor of feat/g39's own tip")
    # A TAG literally named like the unit branch, pointing at the OTHER
    # branch's tip (where fix_head DOES live) -- the shadow.
    _git(root, "tag", "feat/g39", other_fix)
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=other_fix)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g39"}}, root)
    _assert(p.returncode == 2, "G39",
            f"a tag shadowing the unit branch's name must not satisfy ancestry, got "
            f"{p.returncode}: {p.stderr}")
    _assert("is not on" in p.stderr, "G39",
            f"deny reason must name the ancestry failure against the REAL branch, not the "
            f"tag's target: {p.stderr!r}")


def fixture_g40_branch_shadowing_fix_head_prefix_denies() -> None:
    """Round-4 adversarial reproducer: a BRANCH literally named like
    `fix_head`'s own hex prefix, pointing at an UNRELATED commit, must not
    let a bare `<hex>^{commit}` lookup resolve to the BRANCH's tip instead
    of the short-sha object it names. Measured directly (git 2.50.1): with
    a branch literally named the same 12-hex prefix as a real commit's own
    sha, `git rev-parse --verify <prefix>^{commit}` prints `warning: refname
    '<prefix>' is ambiguous.` to stderr and resolves to the BRANCH's tip
    (not the abbreviated object), exit 0 — silently a DIFFERENT commit than
    the one named. RED against 7ed0db7d's patch (no check that the resolved
    sha actually STARTS WITH the given hex); GREEN once that startswith
    check lands."""
    root = _temp_repo("feat/g40")
    row = _write_block_row(root, "feat/g40", "a1", "adversarial-audit",
                            ["a.py:1", "b.py:2"], ["a.py:1"])
    fix_head = _commit_fix(root, "d.py")
    prefix = fix_head[:12]
    # A branch literally named like fix_head's own hex prefix, pointing at
    # an entirely unrelated commit (the BLOCK row's own head_sha).
    _git(root, "branch", prefix, row["head_sha"])
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "fixed"},
                        probe=["c.py:9", "d.py:4"], fix_head=prefix)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/g40"}}, root)
    _assert(p.returncode == 2, "G40",
            f"a branch shadowing fix_head's own hex prefix must deny, got {p.returncode}: {p.stderr}")
    _assert("shadow" in p.stderr, "G40",
            f"deny reason must name the shadow, not merely a generic ancestry failure: {p.stderr!r}")


_G20_28_FIXTURES = [
    ("G20", fixture_g20_no_fix_head_denies),
    ("G21", fixture_g21_fix_head_equals_block_sha_denies),
    ("G22", fixture_g22_relay_unit_branch_name_binds_reachability),
    ("G23", fixture_g23_no_probe_names_fix_changed_denies),
    ("G24", fixture_g24_probe_names_fix_changed_finding_file_allows),
    ("G25", fixture_g25_amend_sibling_fix_allows_orphan_denies),
    ("G26", fixture_g26_claude_project_dir_unset_denies),
    ("G27", fixture_g27_git_timeout_denies),
    ("G28", fixture_g28_real_e1_corpus),
    ("G29", fixture_g29_first_dispatch_stays_git_free_with_hung_shim),
    ("G30", fixture_g30_env_precedence_over_cwd),
    ("G31", fixture_g31_unbound_row_never_satisfiable_by_any_relay_branch),
    ("G32", fixture_g32_round2_no_cross_type_clearing),
    ("G33", fixture_g33_two_units_denied_order_a),
    ("G34", fixture_g34_two_units_denied_order_b),
    ("G35", fixture_g35_argv_boundary_head_sha_option_shaped),
    ("G36", fixture_g36_orphaned_sha_reachable_from_no_ref_denies),
    ("G37", fixture_g37_sha_on_unrelated_branch_denies),
    ("G38", fixture_g38_unresolvable_fix_head_deny_reason_includes_git_stderr),
    ("G39", fixture_g39_tag_shadowing_unit_branch_denies),
    ("G40", fixture_g40_branch_shadowing_fix_head_prefix_denies),
]


def _run_g20_28_fixture_list(fixtures: list[tuple[str, object]]) -> list[str]:
    """The G20-40 arm's own execution+guard logic, factored out so the
    `ran_any` guard can be exercised directly against a MUTATED (here:
    emptied) fixture list, not merely restated in prose — proving the guard
    is live, not dead code that can never observe a False (esc-097
    advisory fix)."""
    failures: list[str] = []
    ran_any = False
    for name, fn in fixtures:
        ran_any = True
        try:
            fn()
            print(f"check-lead-gate[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-lead-gate[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-lead-gate[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)
    if not ran_any:
        # The guard's own self-check: the marker is present but the arm
        # still didn't run any fixture — that is THIS GUARD failing, not a
        # legitimate skip, and must not exit 0.
        failures.append("G20-40 guard: RELAY_R3 marker present but the arm ran no fixtures — guard is broken")
    return failures


def _g20_28_arm() -> tuple[list[str], int]:
    """Returns `(failures, ran_count)` — `ran_count` is 0 when SKIPPED, else
    `len(_G20_28_FIXTURES)`, so `self_test()`'s own final tally counts this
    arm's fixtures when (and only when) it actually ran them, rather than a
    hardcoded `+1` that silently ignores whether G20-40 ran at all.

    esc-097's own fixtures are RED against the current (unpatched) lib —
    the patch lives only in the tracked patch files under
    `docs/plans/63-how-well/proposals/esc-097/` (`.claude/hooks/**` is
    agent-write-denied, so this repo cannot apply them directly). This
    guard makes `--self-test` exit 0 in THIS tree by reporting the arm
    SKIPPED — but it is itself an assertion: if `RELAY_R3` IS present (the
    patch files have been applied, e.g. to a throwaway `cp -R` copy of the
    tree) and the arm STILL reports skipped, that is a bug in this guard,
    and it fails loudly (exit 1 territory) rather than silently
    green-washing a broken check."""
    mod = _lib_module()
    patched = hasattr(mod, "RELAY_R3")
    if not patched:
        print("check-lead-gate[G20-40]: SKIPPED — hook patch esc-097 not applied "
              "(.claude/hooks/lead-gate-lib.py carries no RELAY_R3 marker)")
        return [], 0
    failures = _run_g20_28_fixture_list(_G20_28_FIXTURES)
    # esc-097 (advisory fix): prove the `ran_any` guard above is LIVE — an
    # EMPTY fixture list, run through the IDENTICAL code path, must trip the
    # SAME "ran no fixtures" failure. This is a real mutation test on
    # `_G20_28_FIXTURES` (an empty list), not a restatement of the guard's
    # own logic.
    empty_result = _run_g20_28_fixture_list([])
    if not any("ran no fixtures" in f for f in empty_result):
        failures.append("ran_any guard self-check: an empty fixture list must trip the "
                         "'ran no fixtures' failure via _run_g20_28_fixture_list — it did not")
    else:
        print("check-lead-gate[G20-40 guard self-check]: OK (empty-list mutation correctly fails)")
    return failures, len(_G20_28_FIXTURES)


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
    ("G13", fixture_g13_cross_type_pass_irrelevant_only_same_type_relay_governs),
    ("G14", fixture_g14_unparseable_row_gates_like_block),
    ("G15", fixture_g15_whole_token_anchors_never_raw_substrings),
    ("G16", fixture_g16_reactive_relay_rejected_when_enumeration_present),
    ("G17", fixture_g17_probed_relay_accepted_when_enumeration_present),
    ("G18", fixture_g18_probe_boundary_cases),
    ("G19", fixture_g19_coverage_arm_selected_by_data_not_flag),
    ("OQ1", fixture_oq1_no_open_question_denies),
    ("OQ2", fixture_oq2_open_question_present_allows),
    ("UC1", fixture_uc1_missing_claim_denied),
    ("UC2", fixture_uc2_tested_claim_with_matching_hash_allows),
    ("UC3", fixture_uc3_hash_mismatch_denied),
    ("UC4", fixture_uc4_write_verb_denied),
    ("UC5", fixture_uc5_uncovered_with_reason_allows),
    ("UC6", fixture_uc6_duplicate_uncovered_reason_denied),
    ("UC7", fixture_uc7_no_claim_shaped_line_is_a_noop),
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

    # esc-097 (R3): SKIPPED (exit 0 for this arm only) until a human applies
    # the proposal's hook patch — see `_g20_28_arm`'s own docstring.
    g20_35_failures, g20_35_ran = _g20_28_arm()
    failures.extend(g20_35_failures)

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
    # +1 is N7 (wall-time); the G20-40 arm's own count is 0 when SKIPPED so
    # this total is honest either way (D5: "counts the G arm when it runs").
    print(f"check-lead-gate: all {len(FIXTURES) + 1 + g20_35_ran} self-test fixture(s) passed.")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    print("check_lead_gate.py: usage: --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
