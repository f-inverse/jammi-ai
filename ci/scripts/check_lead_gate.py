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
  OQ1/OQ2 (esc-lead-gate-R10, "open_question") are RETIRED — see R12P*/R12D1/
      R12R2*/R12M1 below, esc-lead-gate-R12's replacement.
  R12P1-R12P12(b), R12D1, R12R2a-d, R12M1  esc-lead-gate-R12 fix round 1
      ("ANTICIPATE BEFORE THE FIX", M1'-M6'): an implementer-type dispatch
      (or a `general-purpose`/`claude`/`fork`/`doc-updater` dispatch that
      DOES name a unit) onto a unit branch with ANY open second-round
      BLOCK is denied unless a `.jammi/gate-state/<slug>.anticipation.
      <tip_sha>.json` artifact exists AT THE BRANCH'S CURRENT TIP, covering
      the UNION of every such open block's own derived keys reduced to one
      entry PER FILE, whose commands the hook re-executes and hash-matches
      in a CLEAN, tip-matching worktree, with at least one execution-class
      command. R12P1/R12P1b/R12P1c: the nine IMPLEMENTER_TYPES require a
      `unit:` line (denied if absent); the four extra GATED types
      (`general-purpose`/`claude`/`fork`/`doc-updater`) do not, but are
      gated identically once a unit IS named. R12P2/R12P2b: no open block
      allows; an unresolvable branch denies. R12P3-R12P10b: missing
      artifact / missing per-file key / stale (pre-tip-move) artifact /
      dirty worktree / hash mismatch / vacuous rc / templated (command,
      hash) reuse / denylisted command / a complete artifact allows / an
      inspector-only artifact denies. R12P11/R12P11b: THE CORE FIX-ROUND-1
      FIX — two open second-round BLOCKs of different types at DIFFERENT
      shas are satisfiable by ONE artifact at the current tip covering
      their UNION (never "newest block only", which would let a
      lead-provoked BLOCK retire an older, unrelated one's obligation for
      free). R12P12: attack commands run in the branch's OWN
      `git worktree list`-resolved worktree, unconfused by a decoy linked
      worktree on another branch. R12D1: the relaxed denylist admits ONLY
      `sh|bash <real path>`, still denies `bash -c`/`curl|sh`/every other
      R11 entry. R12R2a-h: reader 2's `attacks_post` differential — missing
      / no measured difference / a real differential allows / a `.sh` /
      `.yml` new-surface hunk widens the required set by FILE / partial
      coverage denies / a non-object entry denies / a differing post-fix
      command denies. R12M1: a relay still carrying the retired
      `open_question` with no `attacks_post` is denied, naming the
      migration. R12E1-E5: the empty-derived-set path
      (`_r12_empty_set_rejection`) — fewer than 2 lead-chosen keys / a key
      outside the changed-file set / all-inspector / two valid
      execution-class keys allows / an unresolvable main-or-master denies.
      R12F1-F5: a forged `pre_fix_sha` / a mismatched `unit_branch` field /
      a missing `residual_risk` / a non-object JSON artifact / a missing
      `attacks` object, each denies. R12P13-P15: the relaxed denylist's
      remaining arms (a nonexistent bash path, a denied git subcommand, `find
      -delete`), exercised through the REAL dispatch path (never merely
      R12D1's direct, uncached-module-blind call). R12witness/R12reduce/
      R12unparse: the stderr-distinct witness collision, per-file key
      reduction (two raw keys on one file need one entry), and an
      unparseable key mapping to itself while still being required.
      R12timeout/R12alarm: settings.json's PreToolUse timeout pins above
      the hook's own self-alarm and below the harness default; the
      self-alarm itself denies once armed. `--r12-sweep` (its own
      `swarm.yml` step): the M5' AST-based deny-coverage sweep, now over
      EVERY function in the `# R12-BEGIN`/`# R12-END` sentinel region (20
      as of fix round 2, not merely the four core mechanism helpers fix
      round 1 introduced) — see its own docstring for the documented scope
      limit (the dispatch-routing arms `_decide_implementer_dispatch`
      added for M6' sit outside the sentinel region and are not swept).
  esc-lead-gate-R12 fix round 2 (closing audit findings F1-F6 at 5b9fa369,
      plus acceptance folds): F1 (R12F1a/R12F1b) — reader 2 finds the
      pre-fix artifact by scanning for the one whose OWN `covers` list
      names the BLOCK's `ts` (`_r12_find_pre_fix_artifact`), never by
      re-deriving the filename from the BLOCK row's own `head_sha` (which
      only ever names ONE covering artifact when exactly one block was
      open); a genuinely missing covering artifact still denies (`pre_by_
      file` is independently derived from the row's own class_enumeration/
      finding_locations, never forced empty by a missing file). F2
      (R12P10c/R12E7) — `_R12_EXECUTION_PROGRAMS` is now an actual POSITIVE
      membership test (`_r12_is_execution_class`): a `printf`/`echo`
      placeholder is neither inspector-class nor execution-class and
      denies on both the by-file and empty-set paths, where before it
      silently allowed (it was declared but never checked). F3 (R12R2j) —
      reader 2's required set widens by EVERY file `git diff --name-only
      block_sha..fix_head` names, not merely `keys(artifact) ∪
      new_surfaces`. F6 — the killed "tip equals the BLOCK's own head_sha"
      wording is corrected everywhere it survived (SELF-FAILURE-MODES.md,
      the escapes.jsonl row, and this file's own R12 section header).
      Acceptance folds: R12witness is relabeled a STANDING PIN (never
      retired); R12norm — no R12 deny-return literal ever spells the
      standalone word "rm" (a STATIC AST check over the sentinel region,
      not a sample of one runtime message); R12phase — a slow attack does
      not exhaust the following git-call phase's own freshly-minted
      budget; R12residual — the sweep's own `# R12-RESIDUAL: <reason>`
      marker parsing; R12sweepmeta — a meta-test proving the sweep
      mechanism itself flags a deliberately-injected, genuinely
      unreachable canary arm as a SURVIVOR (run only from `--r12-sweep`,
      never `--self-test` — like the real sweep, it drives the same slow
      mutation/re-run pipeline). Item 9: `test_check_lead_gate.py`'s
      subprocess timeout raised to a measured >=2x margin (150s) over the
      64.8-72.7s the R12 fixture set now takes. Item 11:
      `ci/scripts/derive_r12_grandfather.py` derives the grandfather list's
      real content from `docs/rigor/*.jsonl` + `.jammi/gate-state/*.jsonl`
      at merge time (never guessed by hand). Item 12: `_witness_hash` runs
      `stdout` through `_r12_normalize_output` first (R12norm2) — an
      execution-class command's own wall-clock timing text (a cargo/
      libtest "finished in X.XXs" summary line) must never be the thing
      that makes reader 2's pre/post differential "pass".
  esc-lead-gate-R12 fix round 3, item 8 (the fold-9 replacement).
      8a (R12G1-R12G5): `ci/lead-gate-required-commands.txt` is committed,
      human-amend-only, one command per line with a measured runtime
      annotation. Reader 1 (`_r12_gates_shape_rejection`, `judge_rc=False`)
      requires the pre-fix anticipation artifact's own `gates` object to
      name every committed line VERBATIM with an integer `rc` — SHAPE
      only, the VALUE never judged (the pre-fix tip is expected broken).
      Reader 2 (`judge_rc=True`) requires the relay's own `gates` to ALSO
      show `rc == 0` for every line. Reader 3 (`check_rigor_record.py`'s
      `check_required_gates`, RR14-RR16) fails the COMMITTED anticipation
      export's LATEST row on a missing line or a non-zero `rc` — never
      re-executes the gates (already CI jobs elsewhere). 8b (R12M8b1-
      R12M8b5): `_mutations_rejection`, armed by the DATA — the fix's own
      diff adds a new definition in a file the BLOCK's own
      finding_locations also names — requires the relay's `mutations`
      array to carry 1-3 LABELED-sample rows, each either an ACCEPTED
      mutation (`rc_before==0 ∧ rc_after!=0 ∧ marker_after` names a
      committed TEST-failure marker) or an explicit `uncovered` reason.
      8c (R12X1-R12X3): `_exclusions_rejection`, armed by the fix's diff
      adding a NEW TEST definition (`_r12_new_test_surfaces`, a heuristic
      filter over `_parse_new_surfaces`) — requires the relay's own
      `exclusions` object to name, for each, a non-empty case the attack
      does NOT cover, normalized-distinct from its siblings and from the
      unit's own previous relay of the same agent_type.
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
R12_SWEEP_SURVIVORS_PATH = REPO_ROOT / "ci" / "scripts" / "r12_sweep_survivors.txt"

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
# Fix round 5 Z4: `ci/lead-gate-required-commands.txt` missing/empty is now
# a hard DENY in reader 1/2 (never a silent "no obligation"), so `_temp_repo`
# commits a baseline, never-executed line by this name on EVERY fixture's
# seed commit; `_write_anticipation_exact`/`_write_relay_exact` default
# `gates` to a matching single-entry `rc: 0` object (the `_GATES_AUTO`
# sentinel below, distinct from an explicit `gates=None`, which still means
# "omit the `gates` key entirely" — the handful of item-8a fixtures that
# test THAT specific arm pass `gates=None` explicitly).
_BASELINE_REQUIRED_COMMAND = "python3 ci/scripts/check_swarm_bijection.py"
_GATES_AUTO = object()

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
    # Fix round 5 Z4: `ci/lead-gate-required-commands.txt` missing/empty is
    # now a hard DENY in reader 1/2 (never a silent "no obligation") — every
    # fixture gets a baseline, non-empty file by default so unrelated
    # fixtures never incidentally exercise item 8a's own DENY arm; a
    # dedicated fixture that wants the missing/empty/all-comment case
    # `unlink()`s or overwrites this file itself.
    (root / "ci").mkdir(parents=True, exist_ok=True)
    (root / "ci" / "lead-gate-required-commands.txt").write_text(
        f"{_BASELINE_REQUIRED_COMMAND}  # measured ~0.0s\n")
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


# esc-lead-gate-R10 is RETIRED by esc-lead-gate-R12 v2 — `open_question` is
# no longer read by any gate arm at all (only `_post_fix_attacks_rejection`'s
# own migration message still looks for it, and only to redirect toward
# `attacks_post`). No fixture-only backward-compat default is needed for it
# any more; `open_question=None` (never written into the artifact) is now
# the ordinary default, and OQ1/OQ2 (which tested the now-dead R10 arm) are
# removed — see the R12 fixtures below for its replacement's coverage.


def _r12_mod():
    """The real `lead-gate-lib.py` module, dynamically loaded and cached —
    the SAME cache key `_relay_path_exact` already uses, so both never
    load two independent copies."""
    import importlib.util
    if "lead_gate_lib_v3" in sys.modules:
        return sys.modules["lead_gate_lib_v3"]
    spec = importlib.util.spec_from_file_location("lead_gate_lib_v3", LEAD_GATE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    sys.modules["lead_gate_lib_v3"] = mod
    return mod


def _r12_hash(rc: int, stdout: str, stderr_line: str = "") -> str:
    return _r12_mod()._witness_hash(rc, stdout, stderr_line)


def _anticipation_path_exact(root: Path, unit_branch: str, tip_sha: str) -> Path:
    """M1' (fix round 1): the artifact is keyed by the branch's CURRENT
    TIP, never a block's own `ts` — `tip_sha` here is normally the value
    `_write_block_row` returned as `row["head_sha"]` at the moment it was
    written (the branch's tip has not moved since, in every fixture that
    does not deliberately advance it afterward)."""
    mod = _r12_mod()
    return mod.anticipation_artifact_path(root / ".jammi" / "gate-state", _slug(unit_branch), tip_sha)


def _write_anticipation_exact(root: Path, unit_branch: str, tip_sha: str, attacks: dict,
                               residual_risk: str | None = "fixture residual risk placeholder",
                               covers: list[str] | None = None,
                               gates: dict | None = _GATES_AUTO) -> Path:
    """M1' schema: `{unit_branch, pre_fix_sha, covers, attacks, residual_risk}`
    — `attacks` is keyed PER FILE (`{file: {command, hash}}`), never per
    raw `path:line` finding key. `gates` (esc-lead-gate-R12 fix round 3
    item 8a) DEFAULTS to `_GATES_AUTO` — a single, green entry matching
    `_temp_repo`'s own baseline `ci/lead-gate-required-commands.txt` line
    (fix round 5 Z4 made a missing/incomplete `gates` object a hard DENY
    whenever that file is committed, which it now always is by default) —
    an EXPLICIT `gates=None` still omits the key entirely, for the small
    number of fixtures that test item 8a's own missing-`gates` arm."""
    path = _anticipation_path_exact(root, unit_branch, tip_sha)
    artifact = {"unit_branch": unit_branch, "pre_fix_sha": tip_sha, "attacks": attacks}
    if covers is not None:
        artifact["covers"] = covers
    if residual_risk is not None:
        artifact["residual_risk"] = residual_risk
    if gates is _GATES_AUTO:
        gates = {_BASELINE_REQUIRED_COMMAND: {"rc": 0}}
    if gates is not None:
        artifact["gates"] = gates
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact))
    return path


def _write_required_commands_file(root: Path, commands: list[str]) -> None:
    """esc-lead-gate-R12 fix round 3 item 8a: commits a real
    `ci/lead-gate-required-commands.txt` (the SAME relative path
    `_r12_required_commands_path()` resolves under `CLAUDE_PROJECT_DIR`)
    naming `commands` verbatim, one per line, each with a trailing
    `# measured ~0.0s` annotation -- a fixture-scoped stand-in for the
    real repo's own committed file, never touching it."""
    p = root / "ci" / "lead-gate-required-commands.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{c}  # measured ~0.0s\n" for c in commands))
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add ci/lead-gate-required-commands.txt")


def _auto_r12_attack(key: str) -> dict:
    """A cheap, deterministic, denylist-safe command for a synthetic R12
    file key — `printf` a fixed, key-derived string; no shell
    metacharacters in `key` in this fixture harness."""
    out = f"auto-ok-{key}"
    return {"command": f"printf 'auto-ok-{key}'", "hash": _r12_hash(0, out, "")}


def _auto_r12_baseline_hash(key: str) -> str:
    """A DISTINCT, never-re-executed placeholder pre-fix `hash` for the
    auto-default's pre-fix side — `_post_fix_attacks_rejection` never
    re-runs a pre-fix command (only compares its STORED `hash` against the
    freshly re-executed post-fix `hash`), so this can be any valid-shaped
    value that reliably DIFFERS from `_auto_r12_attack`'s own `hash` —
    guaranteeing the differential always finds a difference for an
    auto-generated key, regardless of whether that key's file happens to
    coincide with one the fix actually changed (G24/G28's own real shape:
    a class_enumeration key naming the SAME file the fix touches)."""
    return _r12_hash(0, f"auto-baseline-{key}", "")


def _write_relay_exact(root: Path, row: dict, sites: dict[str, str] | None = None,
                        probe: list[str] | None = None, fix_head: str | None = None,
                        open_question: str | None = None,
                        claims: dict[str, dict] | None = None,
                        attacks_post: dict | str | None = "auto",
                        override: dict | None = None) -> None:
    """esc-097: `fix_head` (the fix commit's full sha) is written into the
    relay artifact whenever the caller supplies one — the R3 arm's own
    field, always OMITTED unless a caller passes it (so every pre-esc-097
    fixture's relay shape is byte-identical to before).

    esc-lead-gate-R11: `claims` needs no backward-compat default — the arm
    is armed by the DATA (a non-empty hook-derived `claim_sites`), and no
    EXISTING fixture's placeholder fix content matches any claim phrase.

    esc-lead-gate-R12 reader 2 (M4', per FILE): `attacks_post` DEFAULTS to
    the sentinel `"auto"` — every EXISTING call site (none of which
    anticipated R12 at all) keeps satisfying the new, always-armed-by-the-
    DATA `attacks_post` requirement without editing dozens of unrelated
    fixtures, the exact backward-compat shape `fix_head`'s own
    None-means-omitted default already established. `"auto"` derives the
    required FILE set from BOTH `row`'s own `class_enumeration`/
    `finding_locations` (reduced to files via `_key_to_file`, the SAME
    derivation `_r12_required_by_file` performs) AND, when `fix_head` is
    given, every new-surface definition the fix's own `-U0` diff adds
    (`_parse_new_surfaces`, also reduced to files — the UC1-7 fixtures'
    claim-fix content adds a real `def f():` line, which is ALSO a new
    surface under R12's reader-2 widening); auto-writes a MATCHING pre-fix
    anticipation artifact at THIS BLOCK's own `head_sha` (its tip at the
    moment it was written — a fixture that wrote its own via
    `_write_anticipation_exact` is never clobbered) with a cheap,
    reproducible `printf` command per file, and mirrors the SAME command
    into `attacks_post` with a DELIBERATELY DIFFERENT recorded hash
    (`_auto_r12_baseline_hash` vs `_auto_r12_attack`'s own hash —
    `_post_fix_attacks_rejection` never re-executes the PRE-fix command,
    only compares its stored `hash` against the freshly re-executed
    post-fix `hash`, so these two literals differing is enough to satisfy
    the differential UNCONDITIONALLY, regardless of whether an auto file
    happens to coincide with one the fix actually changed — G24/G28's own
    real shape). A fixture that means to test R12 itself passes an
    explicit `attacks_post` dict (or `None` to omit it deliberately)."""
    path = _relay_path_exact(root, row)
    # Fix round 5 Z4: `_relay_rejection` judges the relay's OWN `gates`
    # UNCONDITIONALLY (never scoped to R12-anticipation units) against
    # whatever `ci/lead-gate-required-commands.txt` names — `_temp_repo`'s
    # baseline file default means every relay needs a matching green entry
    # unless a caller's own `override` replaces it (item 8a's own fixtures
    # already do, via `override={"gates": ...}`).
    artifact = {"unit_branch": row["unit_branch"], "agent_type": row["agent_type"], "block_ts": row["ts"],
                "gates": {_BASELINE_REQUIRED_COMMAND: {"rc": 0}}}
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
    if attacks_post == "auto":
        mod = _r12_mod()
        pre_keys = {s for s in (row.get("class_enumeration") or []) if isinstance(s, str)} | \
                   {s for s in (row.get("finding_locations") or []) if isinstance(s, str)}
        pre_files = {mod._key_to_file(k) for k in pre_keys}
        new_surface_files: set[str] = set()
        fix_changed_files: set[str] = set()
        if fix_head is not None and isinstance(row.get("head_sha"), str):
            try:
                # A fixture testing an ADVERSARIAL head_sha/fix_head (e.g.
                # G35/G38) may pass a value this harness-side git call
                # cannot itself resolve — that is the POINT of those
                # fixtures (the hook must deny before ever trusting it,
                # never that this SCAFFOLDING call succeeds). Any failure
                # here just means "no new-surface widening to auto-cover",
                # never a fixture-setup crash.
                diff_out = _git(root, "diff", "-U0", "--end-of-options", row["head_sha"], fix_head)
                new_surface_files = {mod._key_to_file(k) for k in mod._parse_new_surfaces(diff_out).keys()}
                # esc-lead-gate-R12 fix round 2 F3: the real hook widens
                # `required` by EVERY file the fix's own diff changed, not
                # merely new-surface definitions — mirror that here so the
                # dozens of EXISTING fixtures using this "auto" mechanism
                # keep auto-covering whatever the fix touches (including
                # this harness's own marker/placeholder files).
                names_out = _git(root, "diff", "--name-only", "--end-of-options", row["head_sha"], fix_head)
                fix_changed_files = {line for line in names_out.splitlines() if line}
            except Failure:
                new_surface_files = set()
                fix_changed_files = set()
        required = pre_files | new_surface_files | fix_changed_files
        if required:
            apath = _anticipation_path_exact(root, row["unit_branch"], row["head_sha"])
            pre_attacks = json.loads(apath.read_text()).get("attacks", {}) if apath.exists() else {}
            wrote_new = False
            for f in sorted(required):
                if f not in pre_attacks:
                    a = _auto_r12_attack(f)
                    pre_attacks[f] = {"command": a["command"], "hash": _auto_r12_baseline_hash(f)}
                    wrote_new = True
            if wrote_new or not apath.exists():
                _write_anticipation_exact(root, row["unit_branch"], row["head_sha"], pre_attacks)
            post = {}
            for f in sorted(required):
                a = _auto_r12_attack(f)
                cmd = pre_attacks.get(f, {}).get("command") or a["command"]
                post[f] = {"command": cmd, "hash": a["hash"]}
            artifact["attacks_post"] = post
    elif isinstance(attacks_post, dict):
        artifact["attacks_post"] = attacks_post
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


def fixture_r12timeout_settings_pins_above_self_bound() -> None:
    """esc-lead-gate-R12 M1': `.claude/settings.json`'s `lead-gate-pre.sh`
    PreToolUse entry pins an explicit `timeout` ABOVE the hook's own
    in-process self-alarm (`_SELF_ALARM_S`, so the harness never cancels
    the hook — fail-open, output discarded — before the self-alarm has a
    chance to produce a real, logged DENY) and BELOW the documented
    harness default of 600s (a sanity ceiling — a pin far above the
    default provides no protection); `async` is never set (a cancel
    deadline, never a fire-and-forget dispatch)."""
    settings = json.loads(SETTINGS_PATH.read_text())
    pre_entries = settings.get("hooks", {}).get("PreToolUse", [])
    lead_gate_entry = None
    for entry in pre_entries:
        for h in entry.get("hooks", []):
            if Path(h.get("command", "")).name == "lead-gate-pre.sh" or h.get("command", "").endswith("lead-gate-pre.sh"):
                lead_gate_entry = h
    _assert(lead_gate_entry is not None, "R12timeout", "no PreToolUse entry names lead-gate-pre.sh")
    timeout = lead_gate_entry.get("timeout")
    _assert(isinstance(timeout, (int, float)), "R12timeout", f"lead-gate-pre.sh's PreToolUse entry carries no numeric `timeout`: {lead_gate_entry}")
    self_alarm_s = _r12_mod()._SELF_ALARM_S
    _assert(timeout > self_alarm_s, "R12timeout",
            f"settings.json timeout ({timeout}) must exceed the hook's own self-alarm ({self_alarm_s})")
    _assert(timeout < 600, "R12timeout", f"settings.json timeout ({timeout}) must stay below the documented harness default of 600s")
    _assert("async" not in lead_gate_entry, "R12timeout", f"lead-gate-pre.sh's PreToolUse entry must never set `async`: {lead_gate_entry}")


def fixture_r12alarm_self_bound_denies() -> None:
    """esc-lead-gate-R12 M1': `_install_self_alarm`'s `SIGALRM` handler
    denies (exits 2, names the self-bound) if the process is still running
    past `_SELF_ALARM_S` — installs a handler with a near-zero alarm (never
    the real 330s bound, which this fixture cannot afford to wait out) and
    confirms the handler itself fires the documented remedy text, in a
    throwaway subprocess so the real test process is never at risk of the
    alarm firing into it."""
    proc = subprocess.run(
        ["python3", "-c",
         "import importlib.util, signal, time, sys\n"
         f"spec = importlib.util.spec_from_file_location('m', {str(LEAD_GATE_LIB)!r})\n"
         "mod = importlib.util.module_from_spec(spec)\n"
         "spec.loader.exec_module(mod)\n"
         "mod._SELF_ALARM_S = 1\n"
         "mod._install_self_alarm()\n"
         "time.sleep(3)\n"],
        capture_output=True, text=True, timeout=10)
    _assert(proc.returncode == 2, "R12alarm", f"the self-alarm must exit 2 once it fires, got {proc.returncode}: {proc.stderr!r}")
    _assert("self-bound" in proc.stderr, "R12alarm", f"the self-alarm's reason must name the self-bound: {proc.stderr!r}")


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
# OQ1-OQ2 (esc-lead-gate-R10, "open_question") are RETIRED — R10 itself is
# retired, replaced outright by esc-lead-gate-R12 v2's reader 1/2 pre-fix
# anticipation + post-fix differential (below); `open_question` is no
# longer read by any gate arm. See the R12 fixtures below for the
# replacement's coverage, and `_post_fix_attacks_rejection`'s own
# migration-message arm for the ONE place `open_question` is still
# inspected at all (R12M1, below).
# ==========================================================================


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
# R12P1-R12P15, R12E1-R12E7, R12F1-R12F5, R12D1, R12R2-*, R12M1, R12witness,
# R12reduce, R12unparse — esc-lead-gate-R12 v2 ("ANTICIPATE BEFORE THE
# FIX"), fix rounds 1-2. Reader 1 (`_pre_fix_anticipation_rejection`): an
# implementer-type dispatch onto a unit with an open second-round-type
# BLOCK is denied unless a complete, EXECUTED, pre-fix anticipation
# artifact exists at `.jammi/gate-state/<slug>.anticipation.<pre_fix_sha>.
# json` — keyed by the unit branch's CURRENT TIP (`pre_fix_sha`), NEVER by
# any one BLOCK's own `head_sha` (M1': a second, older open BLOCK at a
# DIFFERENT sha would otherwise leave the union's own covering artifact
# unfindable under its own name) — covering the UNION of every open
# second-round BLOCK's derived keys, reduced per file, with that SAME
# `pre_fix_sha` equal to the branch's CURRENT tip and a clean worktree.
# Reader 2 (`_post_fix_attacks_rejection`, the `attacks_post` arm): the
# closing-verifier relay must re-run the SAME per-file keys at `fix_head`
# and show a measured difference for every fix-changed file a pre-fix key
# covers — the covering artifact is found by its OWN `covers` list (F1),
# never by the relay's BLOCK row's own `head_sha`, and the required set
# widens by every file the fix's diff actually changed (F3), not merely
# new-surface definitions.
# ==========================================================================

def _r12_dispatch(root: Path, unit: str, subtype: str = "db") -> "subprocess.CompletedProcess":
    return _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": subtype, "prompt": f"unit: {unit}\nimplement the fix"}}, root)


def fixture_r12p1_no_unit_line_denies_for_implementer_types() -> None:
    """M6': an IMPLEMENTER_TYPES dispatch (e.g. `db`) naming NO unit branch
    at all is now DENIED — every implementer dispatch must say which unit
    it works on. This REPLACES the pre-fix-round-1 "no unit named must
    allow" fixture, which M6' deliberately closes (the fail-open
    enumeration it belonged to)."""
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "db", "prompt": "go fix the thing we discussed"}}, root)
    _assert(p.returncode == 2, "R12P1", f"no unit named must now deny (M6'), got {p.returncode}")
    _assert("requires every implementer dispatch to name the unit" in p.stderr, "R12P1", p.stderr)


def fixture_r12p1b_no_unit_line_allows_for_extra_gated_types() -> None:
    """M6': a `general-purpose`/`claude`/`fork`/`doc-updater` dispatch
    naming NO unit branch stays ALLOWED — these four are generic/harness
    types a lead may dispatch for reasons unrelated to implementing a fix;
    gating an unlabeled dispatch of these would brick the lead."""
    root = _fresh_root()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "general-purpose", "prompt": "go look something up"}}, root)
    _assert(p.returncode == 0, "R12P1b", f"no unit named must allow for an extra-gated type, got {p.returncode}: {p.stderr}")


def fixture_r12p1c_extra_gated_type_armed_when_unit_named() -> None:
    """M6': a `general-purpose` dispatch that DOES name a unit with an open
    verifier-type BLOCK is gated exactly like an IMPLEMENTER_TYPES
    dispatch — closing the fail-open enumeration these four types
    previously sat in unconditionally."""
    unit = "feat/r12p1c"
    root = _temp_repo(unit)
    _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _r12_dispatch(root, unit, subtype="general-purpose")
    _assert(p.returncode == 2, "R12P1c", f"a named unit with an open BLOCK must deny for an extra-gated type too, got {p.returncode}")
    _assert("no anticipation artifact" in p.stderr, "R12P1c", p.stderr)


def fixture_r12p2_reader1_not_armed_no_open_block() -> None:
    """An implementer dispatch naming a real, RESOLVABLE unit branch with
    NO open verifier-type BLOCK on it is allowed — structurally nothing to
    anticipate yet. Uses a real repo (M6' resolves the branch under
    refs/heads/ before checking for an open block)."""
    unit = "feat/r12p2"
    root = _temp_repo(unit)
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12P2", f"no open BLOCK must allow, got {p.returncode}: {p.stderr}")


def fixture_r12p2b_unresolvable_branch_denies() -> None:
    """M6': a `unit:` line naming a branch that does not resolve under
    refs/heads/ DENIES outright — never silently treated as "no unit
    named"."""
    root = _temp_repo("feat/r12p2b-seed")
    p = _r12_dispatch(root, "this-branch-does-not-exist-anywhere")
    _assert(p.returncode == 2, "R12P2b", f"an unresolvable branch must deny, got {p.returncode}")
    _assert("does not resolve under refs/heads/" in p.stderr, "R12P2b", p.stderr)


def fixture_r12p3_missing_artifact_denies() -> None:
    unit = "feat/r12p3"
    root = _temp_repo(unit)
    _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P3", f"a missing anticipation artifact must deny, got {p.returncode}")
    _assert("no anticipation artifact" in p.stderr, "R12P3", f"reason must name it: {p.stderr!r}")


def fixture_r12p4_missing_key_denies() -> None:
    """`a.py:1`/`b.py:2` reduce to two REQUIRED FILES, `a.py`/`b.py` — the
    artifact carries `a.py` but omits `b.py` entirely."""
    unit = "feat/r12p4"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1", "b.py:2"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        # b.py (also required, from class_enumeration's b.py:2) is MISSING.
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P4", f"a missing derived key must deny, got {p.returncode}")
    _assert("omits" in p.stderr, "R12P4", f"reason must name the omitted key(s): {p.stderr!r}")


def fixture_r12p5_tip_moved_since_artifact_recorded_denies() -> None:
    """The anticipation artifact is otherwise complete, but the unit
    branch's CURRENT tip has ALREADY advanced past the artifact's own
    `pre_fix_sha` (a fix commit landed since it was recorded) — M1' keys
    the artifact BY tip, so a stale, pre-move artifact is simply no longer
    found at the CURRENT tip's path at all (never silently reused): the
    non-forgeable ordering claim ("the attacks ran against the tip that is
    now current") cannot be made from a file recorded under a sha that is
    no longer the tip."""
    unit = "feat/r12p5"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    })
    _commit_fix(root, "later.py")  # the branch tip has now moved past pre_fix_sha
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P5", f"a moved tip must deny, got {p.returncode}")
    _assert("no anticipation artifact" in p.stderr, "R12P5",
            f"a stale, pre-move artifact must not be found at the new tip: {p.stderr!r}")


def fixture_r12p5b_dirty_worktree_denies_naming_paths() -> None:
    """M1': ordering evidence ALSO requires the resolved worktree's `git
    status --porcelain` to be EMPTY — an uncommitted, untracked change
    denies, naming the dirty path (never merely "the tip moved")."""
    unit = "feat/r12p5b"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    })
    (root / "uncommitted-scratch.txt").write_text("not committed\n")
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P5b", f"a dirty worktree must deny, got {p.returncode}")
    _assert("uncommitted changes" in p.stderr and "uncommitted-scratch.txt" in p.stderr, "R12P5b", p.stderr)


def fixture_r12p6_hash_mismatch_denies() -> None:
    unit = "feat/r12p6"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "printf hello", "hash": "0" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P6", f"a non-reproducing hash must deny, got {p.returncode}")
    _assert("does not reproduce the recorded hash" in p.stderr, "R12P6", f"{p.stderr!r}")


def fixture_r12p7_vacuous_missing_script_denies() -> None:
    """`rc in (126, 127)` (a missing/non-executable script) DENIES as
    VACUOUS regardless of whether the recorded hash happens to match —
    this proves nothing about the mechanism under attack."""
    unit = "feat/r12p7"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "./this-script-does-not-exist-r12p7.sh", "hash": "a" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P7", f"a vacuous (missing-script) run must deny, got {p.returncode}")
    _assert("VACUOUS" in p.stderr, "R12P7", f"reason must name it VACUOUS: {p.stderr!r}")


def fixture_r12p8_templated_reused_pair_denies() -> None:
    unit = "feat/r12p8"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1", "b.py:2"], ["a.py:1"])
    a = _auto_r12_attack("shared")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "b.py": {"command": a["command"], "hash": a["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P8", f"a reused (command, hash) pair must deny, got {p.returncode}")
    _assert("IDENTICAL" in p.stderr, "R12P8", f"reason must name the templated pair: {p.stderr!r}")


def fixture_r12p9_denylisted_command_denies() -> None:
    """A REAL denylist hit — asserted against the SPECIFIC deny text
    (`is denied:`), never merely the wrapper's own "denied —" prefix (which
    would also appear for an unrelated deny, e.g. a missing artifact, and
    make this fixture pass vacuously without ever exercising the denylist)."""
    unit = "feat/r12p9"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "rm -rf /tmp/should-never-run-r12p9", "hash": "b" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P9", f"a denylisted command must deny, got {p.returncode}")
    _assert("command is denied:" in p.stderr, "R12P9",
            f"the SPECIFIC denylist arm must fire, not merely the wrapper's own text: {p.stderr!r}")
    _assert(not Path("/tmp/should-never-run-r12p9").exists(), "R12P9", "the denied command must never have run")


def fixture_r12p10_complete_artifact_allows() -> None:
    """A complete, EXECUTED anticipation artifact at the branch's CURRENT
    tip, in a CLEAN worktree, ALLOWS the implementer dispatch — proving the
    arm is satisfiable, not a permanent deny. Also proves the RELAXED
    denylist admits `bash <real repo-relative path>` and that a real
    execution-class command (never merely an inspector) satisfies M2'."""
    unit = "feat/r12p10"
    root = _temp_repo(unit)
    (root / "ci").mkdir(parents=True, exist_ok=True)
    (root / "ci" / "probe_r12p10.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe script")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    cmd = "bash ci/probe_r12p10.sh"
    h = _r12_hash(0, "ok", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": cmd, "hash": h},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12P10", f"a complete artifact must allow, got {p.returncode}: {p.stderr}")


def fixture_r12p10b_inspector_only_artifact_denies() -> None:
    """M2': an artifact whose EVERY attack is inspector-class (`cat` here)
    denies — reading a file is not attacking a mechanism; at least one
    attack must be execution-class."""
    unit = "feat/r12p10b"
    root = _temp_repo(unit)
    (root / "a.py").write_text("x = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add a.py")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    h = _r12_hash(0, "x = 1\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "cat a.py", "hash": h},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P10b", f"an inspector-only artifact must deny, got {p.returncode}")
    # The EXACT "carries only inspector-class commands" phrase, never the
    # bare substring "inspector-class" -- the F2 fallback message ALSO
    # contains that substring ("...neither inspector-class nor execution-
    # class..."), so a loose substring check here cannot distinguish the
    # all-inspector arm from its own fallback and would never die under a
    # mutation that neuters ONLY the all-inspector arm specifically.
    _assert("carries only inspector-class commands" in p.stderr, "R12P10b", p.stderr)


def fixture_r12p10c_printf_only_artifact_denies() -> None:
    """esc-lead-gate-R12 fix round 2 F2: a `printf`-only artifact is
    NEITHER inspector-class (it is not in `_R12_INSPECTOR_PROGRAMS`) NOR
    execution-class (`_R12_EXECUTION_PROGRAMS` was declared but never
    actually membership-tested) — before F2 this artifact ALLOWED (the
    `all_inspector` check never fires for a non-inspector placeholder);
    it must now DENY, naming the missing execution-class requirement."""
    unit = "feat/r12p10c"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P10c", f"a printf-only artifact must deny, got {p.returncode}")
    _assert("no execution-class attack" in p.stderr, "R12P10c", p.stderr)


def fixture_r12p11_two_open_blocks_different_shas_still_satisfiable() -> None:
    """THE CORE FIX-ROUND-1 BUG: two open second-round BLOCKs of DIFFERENT
    types, at DIFFERENT shas (an older adversarial-audit, then a NEWER
    acceptance-verifier after a fix commit landed in between) — under the
    fix-round-1 shape (tip must equal EACH block's own head_sha
    individually) this is a PERMANENT deny (the tip cannot equal two
    different shas at once). Under M1' (one artifact at the CURRENT tip,
    covering the UNION of every open block's keys), this is satisfiable."""
    unit = "feat/r12p11"
    root = _temp_repo(unit)
    # esc-lead-gate-R12 fix round 2 F2: at least one attack in the artifact
    # must be execution-class, not merely a printf placeholder — a real
    # tracked probe script, committed BEFORE either BLOCK row so it is
    # already present at both blocks' own head_sha.
    (root / "probe_r12p11.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe_r12p11.sh")
    audit_row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _commit_fix(root, "unrelated.py")  # tip advances past the audit BLOCK's own head_sha
    accept_row = _write_block_row(root, unit, "a2", "acceptance-verifier", ["b.py:2"], ["b.py:2"])
    _assert(audit_row["head_sha"] != accept_row["head_sha"], "R12P11 setup",
            "the two BLOCKs must sit at DIFFERENT shas for this to test anything")
    a = {"command": "bash probe_r12p11.sh", "hash": _r12_hash(0, "ok", "")}
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, accept_row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "b.py": {"command": b["command"], "hash": b["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12P11",
            f"a single artifact at the CURRENT tip covering the union of both blocks' keys "
            f"must allow, got {p.returncode}: {p.stderr}")


def fixture_r12p11b_union_still_requires_the_older_blocks_own_keys() -> None:
    """The SAME two-different-shas setup, but the artifact covers only the
    NEWER block's own key (`b.py`) — proving this is the UNION arm, never
    "the newest block only" (which would let a lead-provoked
    acceptance-verifier BLOCK retire an older, unrelated audit's
    obligation for free)."""
    unit = "feat/r12p11b"
    root = _temp_repo(unit)
    _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _commit_fix(root, "unrelated.py")
    accept_row = _write_block_row(root, unit, "a2", "acceptance-verifier", ["b.py:2"], ["b.py:2"])
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, accept_row["head_sha"], {
        "b.py": {"command": b["command"], "hash": b["hash"]},
        # a.py (the OLDER, still-open adversarial-audit's own key) is
        # deliberately MISSING.
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P11b",
            f"covering only the newest block must still deny — the older block's own key is "
            f"required too, got {p.returncode}")
    _assert("omits" in p.stderr and "a.py" in p.stderr, "R12P11b", p.stderr)


def fixture_r12p12_computed_cwd_not_project_dir() -> None:
    """Attack commands run in the `git worktree list`-resolved worktree for
    the unit branch. `root` (checked out on `unit_branch`, where the BLOCK
    row and the anticipation artifact both live, and where
    `CLAUDE_PROJECT_DIR` points for this dispatch) carries a file a SECOND,
    decoy linked worktree (checked out on an unrelated branch, added
    AFTER) does not — proving the resolution is BY BRANCH NAME via `git
    worktree list --porcelain`, never "whichever worktree happens to sit
    at a fixed relative path", and never confused by the decoy's presence."""
    unit = "feat/r12p12"
    root = _temp_repo(unit)
    (root / "only-here.sh").write_text("#!/bin/sh\nprintf present\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add only-here.sh")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    cmd = "bash only-here.sh"
    h = _r12_hash(0, "present", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": cmd, "hash": h},
    })
    decoy_dir = _fresh_root() / "wt"
    _git(root, "worktree", "add", "-q", "-b", f"r12p12-decoy-branch-{id(root)}", str(decoy_dir), "HEAD~2")
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12P12",
            f"the attack must run in the branch's OWN resolved worktree, unconfused by a decoy "
            f"linked worktree on another branch, got {p.returncode}: {p.stderr}")


def fixture_r12p13_bash_nonexistent_path_denies_via_dispatch() -> None:
    """The `_r12_attack_command_denied` "does not resolve to a real,
    existing file" arm, exercised through the REAL subprocess dispatch
    path (never merely the direct in-process call R12D1 makes, which loads
    a cached module reference and cannot observe an AST-mutated copy)."""
    unit = "feat/r12p13"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "bash this-does-not-exist-r12p13.sh", "hash": "a" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P13", f"a bash command naming a nonexistent script must deny, got {p.returncode}")
    _assert("does not resolve to a real, existing file" in p.stderr, "R12P13", p.stderr)


def fixture_r12p14_git_subcommand_denied_via_dispatch() -> None:
    unit = "feat/r12p14"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "git reset --hard", "hash": "a" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P14", f"a denied git subcommand must deny, got {p.returncode}")
    _assert("denied git subcommand" in p.stderr, "R12P14", p.stderr)


def fixture_r12p15_find_delete_denied_via_dispatch() -> None:
    unit = "feat/r12p15"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "find . -delete", "hash": "a" * 64},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P15", f"find with -delete must deny, got {p.returncode}")
    _assert("-delete/-exec/-execdir is denied" in p.stderr, "R12P15", p.stderr)


def fixture_r12p16_missing_command_field_denies() -> None:
    """`_r12_validate_and_run_entry`'s own `has no \\`command\\`` determinant,
    exercised through the real dispatch path — an attack entry with NO
    `command` key at all (never merely an empty string, which is the same
    arm but a different literal shape)."""
    unit = "feat/r12p16"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"hash": "a" * 64},  # `command` deliberately omitted
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P16", f"a missing `command` field must deny, got {p.returncode}")
    _assert("has no `command`" in p.stderr, "R12P16", p.stderr)


def fixture_r12p17_invalid_hash_shape_denies() -> None:
    """`_r12_validate_and_run_entry`'s own `has no valid \\`hash\\`` shape
    determinant — a `hash` field that is not 64 hex characters (never the
    SEPARATE "does not reproduce" arm, which requires a syntactically
    valid but wrong hash; this is the field failing `_OUTPUT_HASH_RE`
    outright)."""
    unit = "feat/r12p17"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "printf hello", "hash": "not-a-valid-sha256-hash"},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P17", f"an invalid hash shape must deny, got {p.returncode}")
    _assert("has no valid `hash`" in p.stderr, "R12P17", p.stderr)


def fixture_r12p18_no_linked_worktree_denies() -> None:
    """`_resolve_worktree_cwd`'s own "no linked worktree resolves" arm: the
    dispatched unit branch DOES resolve under `refs/heads/` (a real ref,
    at exactly the tip the anticipation artifact records) — but NO `git
    worktree` entry (linked or the main one) is checked out on it, so an
    attack command has no real files to run against. Built by creating the
    ref to point at the BLOCK row's own `head_sha` AFTER that commit
    exists (never before), so the tip-equality check earlier in the chain
    is satisfied and this specific arm is what actually fires."""
    unit = "feat/r12p18"
    root = _temp_repo(unit)
    orphan = "feat/r12p18-orphan-never-checked-out"
    row = _write_block_row(root, orphan, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _git(root, "branch", orphan, row["head_sha"])
    _write_anticipation_exact(root, orphan, row["head_sha"], {
        "a.py": {"command": "printf ok", "hash": _r12_hash(0, "ok", "")},
    })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "db", "prompt": f"unit: {orphan}\nimplement the fix"}}, root)
    _assert(p.returncode == 2, "R12P18", f"a unit branch with no linked worktree must deny, got {p.returncode}: {p.stderr}")
    _assert("no linked worktree resolves" in p.stderr, "R12P18", p.stderr)


def fixture_r12p19_execution_class_direct_program_name_allows() -> None:
    """`_r12_is_execution_class`'s own cargo/python3/pytest/make branch —
    every OTHER execution-class fixture in this suite uses `bash <path>`;
    this proves the direct program-name match is independently load-
    bearing (an artifact whose ONLY execution-class command is `python3
    -c ...`, never a bash/sh script, still allows)."""
    unit = "feat/r12p19"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    cmd = "python3 -c \"print('r12p19-ok')\""
    h = _r12_hash(0, "r12p19-ok\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": cmd, "hash": h},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12P19", f"a python3-shaped execution-class command must allow, got {p.returncode}: {p.stderr}")


def fixture_r12p20_non_dict_attack_entry_denies() -> None:
    """`_r12_validate_and_run_entry`'s own "is not an object" arm,
    exercised through READER 1's by-file loop (never reader 2's relay
    path, which pre-checks `isinstance(entry, dict)` itself before ever
    delegating — see R12R2g) — an `attacks["a.py"]` value that is a plain
    string, not a `{command, hash}` object."""
    unit = "feat/r12p20"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": "not-an-object",
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12P20", f"a non-object attacks[f] entry must deny, got {p.returncode}: {p.stderr}")
    _assert("is not an object" in p.stderr, "R12P20", p.stderr)


def _r12_empty_set_repo(unit: str) -> Path:
    """A real repo whose unit branch has committed TWO real changes over
    `main` (`a.py`, `b.py`) — the changed-file set an empty-derived-set
    lead-chosen key must name a member of."""
    root = _temp_repo(unit)
    (root / "a.py").write_text("a = 1\n")
    (root / "b.py").write_text("b = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "the unit's own changes: a.py, b.py")
    return root


def fixture_r12e1_empty_derived_set_fewer_than_two_keys_denies() -> None:
    """M2': an `uncertain` BLOCK with NO `finding_locations`/
    `class_enumeration` at all reaches `_r12_empty_set_rejection` — a
    single lead-chosen key is not enough (needs >=2)."""
    unit = "feat/r12e1"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E1", f"fewer than 2 lead-chosen keys must deny, got {p.returncode}")
    _assert("fewer than 2 lead-chosen" in p.stderr, "R12E1", p.stderr)


def fixture_r12e2_empty_derived_set_key_outside_changed_files_denies() -> None:
    """M2': a lead-chosen key naming a file the unit did NOT actually
    change (against `main`) denies, even with >=2 keys total."""
    unit = "feat/r12e2"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    a = _auto_r12_attack("a.py")
    z = _auto_r12_attack("z.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "z.py": {"command": z["command"], "hash": z["hash"]},  # z.py was never touched
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E2", f"a key outside the changed-file set must deny, got {p.returncode}")
    _assert("is not in the unit's own changed-file set" in p.stderr, "R12E2", p.stderr)


def fixture_r12e3_empty_derived_set_all_inspector_denies() -> None:
    """M2': >=2 lead-chosen keys, both naming real changed files, both
    inspector-class (`cat`) — denies naming inspector-only."""
    unit = "feat/r12e3"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    a_hash = _r12_hash(0, "a = 1\n", "")
    b_hash = _r12_hash(0, "b = 1\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "cat a.py", "hash": a_hash},
        "b.py": {"command": "cat b.py", "hash": b_hash},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E3", f"an all-inspector empty-set artifact must deny, got {p.returncode}")
    _assert("carries only inspector-class commands" in p.stderr, "R12E3", p.stderr)


def fixture_r12e4_empty_derived_set_two_valid_execution_class_allows() -> None:
    """M2': the satisfiable case — >=2 lead-chosen keys, both real changed
    files, at least one execution-class (a real script), distinct
    commands/hashes — ALLOWS."""
    unit = "feat/r12e4"
    root = _r12_empty_set_repo(unit)
    (root / "probe.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe.sh")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    h_ok = _r12_hash(0, "ok", "")
    b_hash = _r12_hash(0, "b = 1\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "bash probe.sh", "hash": h_ok},
        "b.py": {"command": "cat b.py", "hash": b_hash},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12E4", f"two valid, changed-file, execution-class-including keys must allow, got {p.returncode}: {p.stderr}")


def fixture_r12e5_empty_derived_set_unresolvable_main_denies() -> None:
    """M2': when neither `main` nor `master` resolves at all (a repo whose
    default branch has neither name), the changed-file set cannot be
    derived — denies naming it, never silently bypassed."""
    unit = "feat/r12e5"
    root = _fresh_root()
    _git(root, "init", "-q", "-b", "trunk")
    _git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text(".jammi/\n")
    (root / "ci").mkdir(parents=True, exist_ok=True)
    (root / "ci" / "lead-gate-required-commands.txt").write_text(
        f"{_BASELINE_REQUIRED_COMMAND}  # measured ~0.0s\n")
    (root / "a.py").write_text("a = 1\n")
    (root / "b.py").write_text("b = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed on trunk")
    _git(root, "checkout", "-q", "-b", unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    a = _auto_r12_attack("a.py")
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "b.py": {"command": b["command"], "hash": b["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E5", f"an unresolvable main/master must deny, got {p.returncode}")
    _assert("could not resolve a merge-base against main/master" in p.stderr, "R12E5", p.stderr)


def fixture_r12e6_empty_derived_set_malformed_entry_denies() -> None:
    """The empty-set path's own `err is not None: return err` propagation
    (distinct from the by-file path's equivalent, R12P6-9) — a malformed
    entry (bad hash) among otherwise-valid lead-chosen keys still denies,
    naming the reproduction failure."""
    unit = "feat/r12e6"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    b_hash = _r12_hash(0, "b = 1\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": "cat a.py", "hash": "0" * 64},  # wrong hash
        "b.py": {"command": "cat b.py", "hash": b_hash},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E6", f"a malformed empty-set entry must deny, got {p.returncode}")
    _assert("does not reproduce the recorded hash" in p.stderr, "R12E6", p.stderr)


def fixture_r12e7_empty_derived_set_printf_only_denies() -> None:
    """esc-lead-gate-R12 fix round 2 F2, empty-set path: two lead-chosen
    keys, both real changed files, both `printf` — neither inspector-class
    nor execution-class. Before F2 this ALLOWED (the `all_inspector` check
    never fires for a non-inspector placeholder); it must now DENY."""
    unit = "feat/r12e7"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    a = _auto_r12_attack("a.py")
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "b.py": {"command": b["command"], "hash": b["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E7", f"a printf-only empty-set artifact must deny, got {p.returncode}")
    _assert("no execution-class attack" in p.stderr, "R12E7", p.stderr)


def fixture_r12e8_empty_derived_set_reused_pair_denies() -> None:
    """fix round 5 (sweep closure): the empty-derived-set path has NO
    shared-validator call downstream of `_r12_validate_and_run_entry`
    (unlike the non-empty by_file path, where R12P8's identical scenario
    is caught redundantly by BOTH `_r12_validate_and_run_entry`'s own
    pair-reuse check and `_r12_anticipation_rejection`'s) -- two
    lead-chosen keys reusing the IDENTICAL (command, hash) pair must
    still deny HERE, with no backstop to catch it if this arm alone were
    silently neutered."""
    unit = "feat/r12e8"
    root = _r12_empty_set_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", None, [])
    shared = _auto_r12_attack("shared")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": shared["command"], "hash": shared["hash"]},
        "b.py": {"command": shared["command"], "hash": shared["hash"]},
    })
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12E8", f"a reused (command, hash) pair on the empty-set path must deny, got {p.returncode}")
    _assert("IDENTICAL" in p.stderr, "R12E8", f"reason must name the templated pair: {p.stderr!r}")


def fixture_r12f1_pre_fix_sha_mismatch_denies() -> None:
    """The artifact is found at the CORRECT tip-keyed filename, but its OWN
    internal `pre_fix_sha` field (a forged or copy-pasted artifact) does
    not match — denies, distinct from the "no artifact" / "tip moved"
    arms."""
    unit = "feat/r12f1"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    path = _anticipation_path_exact(root, unit, row["head_sha"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "unit_branch": unit, "pre_fix_sha": "0" * 40,  # forged, does not match the filename's own tip
        "attacks": {"a.py": {"command": a["command"], "hash": a["hash"]}},
        "residual_risk": "fixture residual",
    }))
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12F1", f"a pre_fix_sha mismatch must deny, got {p.returncode}")
    _assert("does not match its own filename's tip" in p.stderr, "R12F1", p.stderr)


def fixture_r12f2_unit_branch_field_mismatch_denies() -> None:
    unit = "feat/r12f2"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    path = _anticipation_path_exact(root, unit, row["head_sha"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "unit_branch": "feat/some-other-unit", "pre_fix_sha": row["head_sha"],
        "attacks": {"a.py": {"command": a["command"], "hash": a["hash"]}},
        "residual_risk": "fixture residual",
    }))
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12F2", f"a mismatched unit_branch field must deny, got {p.returncode}")
    _assert("`unit_branch` does not name this unit" in p.stderr, "R12F2", p.stderr)


def fixture_r12f3_missing_residual_risk_denies() -> None:
    unit = "feat/r12f3"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"],
                               {"a.py": {"command": a["command"], "hash": a["hash"]}},
                               residual_risk=None)
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12F3", f"a missing residual_risk must deny, got {p.returncode}")
    _assert("carries no non-empty `residual_risk`" in p.stderr, "R12F3", p.stderr)


def fixture_r12f4_non_dict_json_artifact_denies() -> None:
    unit = "feat/r12f4"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    path = _anticipation_path_exact(root, unit, row["head_sha"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["not", "an", "object"]))
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12F4", f"a non-object JSON artifact must deny, got {p.returncode}")
    _assert("is not a JSON object" in p.stderr, "R12F4", p.stderr)


def fixture_r12f5_attacks_field_missing_denies() -> None:
    unit = "feat/r12f5"
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    path = _anticipation_path_exact(root, unit, row["head_sha"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "unit_branch": unit, "pre_fix_sha": row["head_sha"], "residual_risk": "fixture",
        "gates": {_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        # `attacks` deliberately omitted.
    }))
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12F5", f"a missing `attacks` object must deny, got {p.returncode}")
    _assert("carries no `attacks` object" in p.stderr, "R12F5", p.stderr)


def fixture_r12r2f_partial_attacks_post_coverage_denies() -> None:
    """Reader 2: an artifact covering TWO required files, but the relay's
    `attacks_post` covers only ONE — denied, naming the omission (never
    satisfied by partial coverage)."""
    unit = "feat/r12r2f"
    root = _temp_repo(unit)
    (root / "a.py").write_text("a = 1\n")
    (root / "b.py").write_text("b = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed a.py, b.py")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1", "b.py:2"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
        "b.py": {"command": b["command"], "hash": b["hash"]},
    })
    (root / "a.py").write_text("a = 2\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: change a.py only")
    fix_head = _git(root, "rev-parse", "HEAD")
    a_post_hash = _r12_hash(0, "a = 2\n", "")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed", "b.py:2": "n/a"}, probe=["c.py:9", "a.py"],
                        fix_head=fix_head,
                        attacks_post={"a.py": {"command": "cat a.py", "hash": a_post_hash}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2f"}}, root)
    _assert(p.returncode == 2, "R12R2f", f"partial attacks_post coverage must deny, got {p.returncode}")
    _assert("omits" in p.stderr and "required site" in p.stderr, "R12R2f", p.stderr)


def fixture_r12r2g_attacks_post_entry_not_object_denies() -> None:
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2g")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head, attacks_post={"state.txt": "not-an-object"})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2g"}}, root)
    _assert(p.returncode == 2, "R12R2g", f"a non-object attacks_post entry must deny, got {p.returncode}")
    _assert("is not an object" in p.stderr, "R12R2g", p.stderr)


def fixture_r12r2h_attacks_post_command_differs_from_pre_fix_denies() -> None:
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2h")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    different_cmd = "head -c4 state.txt"
    post_hash = _r12_hash(0, "FIXE", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": different_cmd, "hash": post_hash}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2h"}}, root)
    _assert(p.returncode == 2, "R12R2h", f"a differing post-fix command must deny, got {p.returncode}")
    _assert("requires the SAME command" in p.stderr, "R12R2h", p.stderr)


def fixture_r12r2i_attacks_post_hash_mismatch_denies() -> None:
    """Reader 2's own `_r12_validate_and_run_entry` propagation (distinct
    from R12R2b's "identical pre/post hash" differential check, which runs
    AFTER this one) — a recorded `attacks_post` hash that does not
    reproduce at all still denies here."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2i")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": "0" * 64}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2i"}}, root)
    _assert(p.returncode == 2, "R12R2i", f"a non-reproducing attacks_post hash must deny, got {p.returncode}")
    _assert("does not reproduce" in p.stderr, "R12R2i", p.stderr)


def fixture_r12r2j_fix_changed_file_with_no_covering_key_denies() -> None:
    """esc-lead-gate-R12 fix round 2 F3: `required` widens by EVERY file
    the fix's own diff changed (`git diff --name-only block_sha..fix_head`),
    not merely `keys(artifact) ∪ new_surfaces`. The fix here also touches
    `config.toml` — a file no pre-fix key names and that introduces no
    def/fn-shaped new surface at all — before F3 this file was invisible
    to reader 2 entirely; it must now be required and denied when
    uncovered."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2j")
    (root / "state.txt").write_text("FIXED\nv1\n")
    (root / "config.toml").write_text("key = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED, plus config.toml")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2j"}}, root)
    _assert(p.returncode == 2, "R12R2j",
            f"a fix-changed file with no covering key must deny, got {p.returncode}: {p.stderr}")
    _assert("config.toml" in p.stderr, "R12R2j", p.stderr)


def fixture_r12witness_stderr_distinguishes_missing_from_real_failure() -> None:
    """STANDING PIN — never a one-round acceptance check to retire once
    this proposal merges; it stays in the suite for as long as
    `_witness_hash` exists, the same way a regression test for a fixed bug
    outlives the bug. The design's own measured collision: `sha256(rc+
    stdout)` ALONE cannot distinguish a missing script (rc=127, empty
    stdout) from a real rc=1 failure with empty stdout; `_witness_hash`
    includes the first stderr line specifically to fix this. Two IDENTICAL
    missing-script runs still hash identically (by design — this is not a
    nonce)."""
    mod = _r12_mod()
    h_missing_1 = mod._witness_hash(127, "", "sh: 1: ./nope.sh: not found")
    h_missing_2 = mod._witness_hash(127, "", "sh: 1: ./nope.sh: not found")
    h_real_fail = mod._witness_hash(1, "", "")
    _assert(h_missing_1 == h_missing_2, "R12witness", "two identical missing-script witnesses must hash identically")
    _assert(h_missing_1 != h_real_fail, "R12witness", "rc=127 and rc=1 must never collide")


def fixture_r12reduce_two_lines_same_file_one_required_entry() -> None:
    """M1': two raw derived keys on the SAME file (`a.py:1`, `a.py:9`)
    reduce to ONE required entry (`a.py`) — a complete artifact covering
    just that ONE file ALLOWS, proving the per-file reduction actually
    reduces (never one required entry per raw line-key)."""
    unit = "feat/r12reduce"
    root = _temp_repo(unit)
    # esc-lead-gate-R12 fix round 2 F2: the sole required entry must carry
    # an execution-class attack, not a printf placeholder.
    (root / "probe_r12reduce.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe_r12reduce.sh")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1", "a.py:9"], ["a.py:1"])
    a = {"command": "bash probe_r12reduce.sh", "hash": _r12_hash(0, "ok", "")}
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12reduce", f"one file entry must satisfy two same-file raw keys, got {p.returncode}: {p.stderr}")


def fixture_r12unparse_prose_key_maps_to_itself_still_required() -> None:
    """M1'/M2': a `class_enumeration` entry that is prose, not
    `path:line`-shaped, maps to ITSELF as its own "file"
    (`_key_to_file` returns it unchanged) — it is NEVER silently exempted;
    missing, it still denies naming the omission; covered under its OWN
    literal text, it allows."""
    unit = "feat/r12unparse"
    root = _temp_repo(unit)
    # esc-lead-gate-R12 fix round 2 F2: the prose key's own attack (the
    # ONLY entry `by_file` actually requires here) must be execution-class.
    (root / "probe_r12unparse.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe_r12unparse.sh")
    prose_key = "the whole loader stage (nccl.rs)"
    row = _write_block_row(root, unit, "a1", "adversarial-audit", [prose_key], [prose_key])
    # An artifact that EXISTS (at the right tip) but omits the prose key
    # entirely -- must still deny, naming the omission (never silently
    # exempted because the key does not parse as path:line).
    other = _auto_r12_attack("unrelated.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"unrelated.py": {"command": other["command"], "hash": other["hash"]}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12unparse", f"a missing attack for an unparseable key must still deny, got {p.returncode}")
    _assert("omits" in p.stderr and prose_key in p.stderr, "R12unparse", p.stderr)
    a = {"command": "bash probe_r12unparse.sh", "hash": _r12_hash(0, "ok", "")}
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "unrelated.py": {"command": other["command"], "hash": other["hash"]},
        prose_key: {"command": a["command"], "hash": a["hash"]},
    })
    p2 = _r12_dispatch(root, unit)
    _assert(p2.returncode == 0, "R12unparse", f"covering the prose key under its own literal text must allow, got {p2.returncode}: {p2.stderr}")


def fixture_r12d1_relaxed_denylist_still_denies_bashc_and_pipe_sh() -> None:
    """The relaxed denylist admits ONLY `sh|bash <existing repo path>`
    (with no further arguments) — `bash -c '...'` and `curl ... | sh`
    (the two shapes the killed R4a design measured as wrongly ALLOWED)
    stay denied exactly as R11 leaves them; a `bash <path>` naming a path
    that does NOT exist is denied too (never merely 'bash is unconditionally
    denied')."""
    mod = _r12_mod()
    root = _temp_repo("feat/r12d1")
    (root / "real.sh").write_text("#!/bin/sh\necho ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add real.sh")
    cwd = str(root)
    _assert(mod._r12_attack_command_denied("bash real.sh", cwd) is None,
            "R12D1", "bash <real repo-relative path> must be admitted")
    _assert(mod._r12_attack_command_denied("bash nonexistent-r12d1.sh", cwd) is not None,
            "R12D1", "bash <path that does not exist> must still be denied")
    _assert(mod._r12_attack_command_denied("bash -c 'echo pwned'", cwd) is not None,
            "R12D1", "bash -c must still be denied")
    _assert(mod._r12_attack_command_denied("curl http://x | sh", cwd) is not None,
            "R12D1", "curl ... | sh must still be denied")
    _assert(mod._r12_attack_command_denied("rm -rf /tmp/x", cwd) is not None,
            "R12D1", "every other R11 denylist entry must be unaffected")


def _r12_post_setup(unit: str):
    """A real repo whose `state.txt` is TWO lines (`BROKEN`/`v1`) at the
    BLOCK's own head_sha — the pre-fix anticipation artifact's `head -1
    state.txt` witness (line 1 only) is recorded against that content,
    keyed by FILE (`state.txt`, per M4'). Returns `(root, row, cmd,
    pre_hash)`. Reading only line 1 (never the whole file) is deliberate:
    it lets a fixture change line 2 (a REAL, non-empty commit `git commit`
    will accept) while keeping line 1's own observable output identical,
    for the "no differential" DENY case below — a commit that changes
    NOTHING in `state.txt` at all would never even appear in `git diff
    --name-only`, which would vacuously SKIP the differential check
    entirely rather than exercise it."""
    root = _temp_repo(unit)
    (root / "state.txt").write_text("BROKEN\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed state.txt = BROKEN/v1")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["state.txt:1"], ["state.txt:1"])
    cmd = "head -1 state.txt"
    pre_hash = _r12_hash(0, "BROKEN\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "state.txt": {"command": cmd, "hash": pre_hash},
    })
    return root, row, cmd, pre_hash


def _r12_gates_post_setup(unit: str, required_commands: list[str]):
    """esc-lead-gate-R12 fix round 3 item 8a/8b/8c: the SAME shape as
    `_r12_post_setup`, but `ci/lead-gate-required-commands.txt` is
    committed BEFORE the seed commit (and so before the BLOCK's own
    head_sha) — the required-commands file must NEVER itself appear in
    the fix's own `block_sha..fix_head` diff, or F3's own widening would
    require covering it as an attacks_post/gates site too, which is not
    what these fixtures mean to exercise."""
    root = _temp_repo(unit)
    if required_commands:
        _write_required_commands_file(root, required_commands)
    (root / "state.txt").write_text("BROKEN\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed state.txt = BROKEN/v1")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["state.txt:1"], ["state.txt:1"])
    cmd = "head -1 state.txt"
    pre_hash = _r12_hash(0, "BROKEN\n", "")
    return root, row, cmd, pre_hash


def _r12f1_two_block_setup(unit: str):
    """esc-lead-gate-R12 fix round 2 F1 setup: TWO open BLOCKs of different
    types at DIFFERENT shas on the same unit (an older adversarial-audit,
    then a newer acceptance-verifier after a fix commit landed in between)
    and ONE covering artifact recorded at the union's own (newer) tip with
    an explicit `covers` list naming BOTH blocks' `ts` — the real shape
    M1' produces once a second block opens. Returns `(root, audit_row,
    accept_row, cmd, pre_hash)`."""
    root = _temp_repo(unit)
    (root / "state.txt").write_text("BROKEN\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed state.txt = BROKEN/v1")
    audit_row = _write_block_row(root, unit, "a1", "adversarial-audit", ["state.txt:1"], ["state.txt:1"])
    _commit_fix(root, "unrelated.py")  # tip advances past the audit BLOCK's own head_sha
    accept_row = _write_block_row(root, unit, "a2", "acceptance-verifier", ["b.py:2"], ["b.py:2"])
    cmd = "head -1 state.txt"
    pre_hash = _r12_hash(0, "BROKEN\n", "")
    b = _auto_r12_attack("b.py")
    _write_anticipation_exact(root, unit, accept_row["head_sha"], {
        "state.txt": {"command": cmd, "hash": pre_hash},
        "b.py": {"command": b["command"], "hash": b["hash"]},
    }, covers=[audit_row["ts"], accept_row["ts"]])
    return root, audit_row, accept_row, cmd, pre_hash


def fixture_r12f1a_reader2_finds_artifact_by_covers_not_block_head_sha() -> None:
    """esc-lead-gate-R12 fix round 2 F1: the OLDER block's own `head_sha`
    names NO artifact file on disk (the covering artifact sits at the
    UNION's newer tip) — before F1, Reader 2 looked the artifact up by
    `row["head_sha"]` directly, found nothing, and silently treated the
    differential as vacuously satisfied (`required_r12` empty), ALLOWING
    the older block's relay through with no check at all. After F1,
    Reader 2 finds the SAME artifact by its `covers` list and still
    catches an unmoved hash."""
    unit = "feat/r12f1a"
    root, audit_row, accept_row, cmd, pre_hash = _r12f1_two_block_setup(unit)
    _assert(audit_row["head_sha"] != accept_row["head_sha"], "R12F1a setup",
            "the two BLOCKs must sit at DIFFERENT shas for this to test anything")
    (root / "state.txt").write_text("BROKEN\nv2\n")  # only line 2 changes; line 1 never moves
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 2 only, line 1 unchanged")
    fix_head = _git(root, "rev-parse", "HEAD")
    b = _auto_r12_attack("b.py")
    # F3 widens the OLDER block's own fix window by EVERY file changed
    # since its head_sha, which here also sweeps up the setup commits that
    # recorded the NEWER block (its marker file) and the tip-advancing
    # `unrelated.py` commit — cover both with harmless auto attacks so the
    # fixture actually reaches the state.txt differential it means to test.
    marker = _auto_r12_attack(".block-acceptance-verifier-BLOCK.marker")
    unrelated = _auto_r12_attack("unrelated.py")
    _write_relay_exact(root, audit_row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={
                            "state.txt": {"command": cmd, "hash": pre_hash},
                            "b.py": {"command": b["command"], "hash": b["hash"]},
                            ".block-acceptance-verifier-BLOCK.marker": {"command": marker["command"], "hash": marker["hash"]},
                            "unrelated.py": {"command": unrelated["command"], "hash": unrelated["hash"]},
                        })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12F1a",
            f"the OLDER block's relay must still be checked against the covering artifact "
            f"(found by `covers`, never by its own head_sha) and deny on an unmoved hash, "
            f"got {p.returncode}: {p.stderr}")
    _assert("did not observably move" in p.stderr, "R12F1a", p.stderr)


def fixture_r12f1b_deleted_artifact_denies_not_allows() -> None:
    """esc-lead-gate-R12 fix round 2 F1: R12R2b's own setup (a real
    differential the fix DOES move), but the covering anticipation
    artifact is DELETED from disk before the relay is checked. Before F1
    this SILENTLY ALLOWED — a missing artifact forced `pre_by_file = {}`,
    which emptied `required_r12` and skipped the whole reader-2 arm
    instead of denying it. After F1, `pre_by_file` is independently
    derived from the BLOCK row's own class_enumeration/finding_locations,
    so a genuinely missing artifact still trips the "no pre-fix
    anticipation artifact found" guard."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12f1b")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}})
    apath = _anticipation_path_exact(root, row["unit_branch"], row["head_sha"])
    apath.unlink()
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12f1b"}}, root)
    _assert(p.returncode == 2, "R12F1b",
            f"a deleted pre-fix anticipation artifact must DENY, never silently allow, "
            f"got {p.returncode}: {p.stderr}")
    _assert("no pre-fix anticipation artifact found" in p.stderr, "R12F1b", p.stderr)


def fixture_r12r2a_missing_attacks_post_denies() -> None:
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2a")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head, attacks_post=None)
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2a"}}, root)
    _assert(p.returncode == 2, "R12R2a", f"a missing attacks_post must deny, got {p.returncode}")
    _assert("attacks_post" in p.stderr, "R12R2a", f"{p.stderr!r}")


def fixture_r12r2b_no_differential_denies() -> None:
    """`attacks_post` reproduces perfectly, but its hash for the ONE FILE
    covering the changed file (`state.txt`) is IDENTICAL to the pre-fix
    `hash` — the fix changed state.txt's SECOND line (a real, non-empty
    commit, so state.txt genuinely appears in `fix_changed`), but the
    attack only reads LINE 1, which never moved — the fix did not
    observably move anything this attack measures."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2b")
    (root / "state.txt").write_text("BROKEN\nv2\n")  # only line 2 changes
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 2 only, line 1 unchanged")
    fix_head = _git(root, "rev-parse", "HEAD")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": pre_hash}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2b"}}, root)
    _assert(p.returncode == 2, "R12R2b", f"an identical pre/post hash must deny, got {p.returncode}")
    _assert("did not observably move" in p.stderr, "R12R2b", f"{p.stderr!r}")


def fixture_r12r2c_real_differential_allows() -> None:
    """The SAME setup as R12R2b, but the fix ACTUALLY changes line 1 (the
    line the attack reads) — the post-fix hash differs from the pre-fix
    one, and the relay ALLOWS."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2c")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2c"}}, root)
    _assert(p.returncode == 0, "R12R2c", f"a real differential must allow, got {p.returncode}: {p.stderr}")


def fixture_r12r2d_sh_hunk_widens_by_file() -> None:
    """M4': a `.sh` hunk widens the required set by FILE only (never by a
    finer definition-level granularity — a shell function has no `def`/
    `fn` shape `_new_surface_def` recognizes anyway, so this also proves
    the file-level fallback actually fires for a real new-surface diff).
    `run.sh` is entirely NEW (no pre-fix key covers it, so no "same
    command as pre-fix" constraint applies); `state.txt` is UNCHANGED by
    this fix and its `attacks_post` entry re-runs the SAME pre-fix
    command, reproducing the SAME hash — no differential violation, since
    `state.txt` never appears in `fix_changed` here."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2d")
    (root / "run.sh").write_text("#!/bin/sh\nfoo() {\n  echo hi\n}\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add run.sh, state.txt unchanged")
    fix_head = _git(root, "rev-parse", "HEAD")
    run_sh_hash = _r12_hash(0, (root / "run.sh").read_text(), "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "run.sh"],
                        fix_head=fix_head,
                        attacks_post={
                            "state.txt": {"command": cmd, "hash": pre_hash},
                            "run.sh": {"command": "cat run.sh", "hash": run_sh_hash},
                        })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2d"}}, root)
    _assert(p.returncode == 0, "R12R2d", f"a new .sh surface widened by file must allow once covered, got {p.returncode}: {p.stderr}")


def fixture_r12r2e_yml_hunk_widens_by_file() -> None:
    """M4': the SAME property as R12R2d, for a `.yml` hunk (`_WORKFLOW_STEP_RE`
    recognizes a new `- name: ...` step) — a SECOND, independent
    extension, never assumed identical-by-inference from the `.sh` case."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12r2e")
    (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    yml_path = root / ".github" / "workflows" / "new.yml"
    yml_path.write_text("jobs:\n  x:\n    steps:\n      - name: a new step\n        run: echo hi\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add .github/workflows/new.yml, state.txt unchanged")
    fix_head = _git(root, "rev-parse", "HEAD")
    yml_hash = _r12_hash(0, yml_path.read_text(), "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", ".github/workflows/new.yml"],
                        fix_head=fix_head,
                        attacks_post={
                            "state.txt": {"command": cmd, "hash": pre_hash},
                            ".github/workflows/new.yml": {"command": "cat .github/workflows/new.yml", "hash": yml_hash},
                        })
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12r2e"}}, root)
    _assert(p.returncode == 0, "R12R2e", f"a new .yml surface widened by file must allow once covered, got {p.returncode}: {p.stderr}")


def fixture_r12m1_open_question_without_attacks_post_denies() -> None:
    """A relay still carrying the RETIRED `open_question` field, with no
    `attacks_post`, is denied with a message naming the replacement —
    never silently accepted under the old schema."""
    root, row, cmd, pre_hash = _r12_post_setup("feat/r12m1")
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head, attacks_post=None,
                        open_question="qux.py:3 — next round should check nccl.rs")
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12m1"}}, root)
    _assert(p.returncode == 2, "R12M1", f"open_question with no attacks_post must deny, got {p.returncode}")
    _assert("REPLACES" in p.stderr and "attacks_post" in p.stderr, "R12M1",
            f"reason must name the migration to attacks_post: {p.stderr!r}")


_R12_RM_WORD_RE = re.compile(r"\brm\b", re.IGNORECASE)


def _r12_return_string_literals(source: str, begin: int, end: int) -> list[tuple[int, str]]:
    """`[(lineno, literal_text_chunk), ...]` for every literal STRING chunk
    that could reach a caller from a `Return` inside the R12 sentinel line
    range — f-string literal segments (never the interpolated `{...}`
    expressions, which are not authored prose), plain string constants,
    and `+`-concatenated/tuple/list-wrapped combinations of the above."""
    tree = ast.parse(source)
    out: list[tuple[int, str]] = []

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    out.append((v.lineno, v.value))
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                collect(elt)
        elif isinstance(node, ast.BinOp):
            collect(node.left)
            collect(node.right)

    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and begin <= node.lineno <= end and node.value is not None:
            collect(node.value)
    return out


def fixture_r12norm_no_rm_in_any_deny_text() -> None:
    """esc-lead-gate-R12 fix round 2 acceptance fold: "no R12 deny message
    ever names the `rm` hatch" is a STATIC property over every literal
    string chunk a `Return` inside the `# R12-BEGIN`/`# R12-END` sentinel
    region could ever emit — never a sample of one runtime message. Word-
    bounded (`\\brm\\b`) so "alarm"/"confirm"/"warm"/"term" never false-
    positive; a denylisted `rm` COMMAND legitimately appears in fixture
    DATA elsewhere (R12P9) — this checks only what the HOOK ITSELF prints,
    never what a lead-supplied command string might contain."""
    source = LEAD_GATE_LIB.read_text()
    begin, end = _r12_sentinel_line_range(source)
    hits = [(ln, text) for ln, text in _r12_return_string_literals(source, begin, end)
            if _R12_RM_WORD_RE.search(text)]
    _assert(not hits, "R12norm",
            f"an R12 deny-return literal spells the standalone word 'rm' — even a message that "
            f"DISCLAIMS the rm hatch still names it: {hits}")


def fixture_r12phase_slow_attack_does_not_exhaust_next_git_phase() -> None:
    """M1'/fix round 2 acceptance fold: the per-phase deadline is minted
    FRESH for each phase (ref/worktree resolution, then attacks, then any
    LATER git call) — a slow attack command must not eat into the budget
    of the git-call phase that follows it. A single ~1s attack (well under
    `_ATTACK_BUDGET_S`, generous margin against timing jitter) followed by
    the hook's OWN subsequent git calls (worktree HEAD re-check inside
    Reader 1) must still complete well inside a single fresh
    `_GIT_BUDGET_S` window — i.e. the whole dispatch still returns quickly,
    never stalling for anywhere near the attack's own duration stacked
    onto a git budget that should already have been spent and re-minted."""
    unit = "feat/r12phase"
    root = _temp_repo(unit)
    (root / "slow_probe_r12phase.sh").write_text("#!/bin/sh\nsleep 1\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add slow_probe_r12phase.sh")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    cmd = "bash slow_probe_r12phase.sh"
    h = _r12_hash(0, "ok", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": cmd, "hash": h},
    })
    start = time.monotonic()
    p = _r12_dispatch(root, unit)
    elapsed = time.monotonic() - start
    _assert(p.returncode == 0, "R12phase", f"a real ~1s attack must still allow, got {p.returncode}: {p.stderr}")
    # Generous margin: the attack itself takes ~1s; the WHOLE dispatch
    # (attack + every git phase around it) must not balloon anywhere near
    # a shared/starved-budget shape — 10s is comfortably above the ~1s
    # attack plus ordinary git-call overhead, comfortably below any
    # budget that would indicate the NEXT phase inherited the attack's
    # own exhausted clock.
    _assert(elapsed < 10.0, "R12phase",
            f"the dispatch took {elapsed:.2f}s — the attack phase must not exhaust the "
            "following git-call phase's own freshly-minted budget")


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


# ==========================================================================
# esc-lead-gate-R12 M5' — the deny-coverage sweep is COMMITTED. Enumerates
# every deny-Return inside the `# R12-BEGIN`/`# R12-END` sentinel region of
# `lead-gate-lib.py` (the four `str | None`-returning core mechanism
# helpers R12 fix round 1 introduced) by AST, neuters each arm ALONE
# (its nearest enclosing `if` test forced to `ast.Constant(False)`, so the
# guard never fires and that specific deny becomes unreachable), imports
# the mutated source into a throwaway hooks directory, and re-runs every
# registered "R12*"-named fixture against it — FAILING when an arm's
# neutering kills NO fixture (the arm has no oracle proving it fires).
# ==========================================================================

import ast  # noqa: E402  (kept local to this section, mirrors the module's own late imports)

_R12_SWEEP_FUNCS = {
    # esc-lead-gate-R12 fix round 5 Z6: this set is EQUAL to the top-level
    # `FunctionDef` set of the `# R12-BEGIN`/`# R12-END` sentinel region
    # (nested defs excluded) — a self-test fixture (`R12sweepast`) asserts
    # that equality BY AST against the real file, so this comment names no
    # count that can go stale: a function added to the region and omitted
    # here fails `--self-test` immediately, never merely "was covered by
    # fewer fixtures than it should have been." A function with no
    # CONDITIONAL non-`None`/non-fallthrough return contributes ZERO
    # positions automatically — `_r12_deny_if_positions` only counts a
    # `Return` inside an `If` — so listing every such function here too is
    # harmless and keeps this set a straightforward, auditable ENUMERATION
    # of the region rather than a hand-picked subset.
    "_install_self_alarm", "_new_attack_deadline", "_derived_attack_keys",
    "_key_to_file", "anticipation_artifact_path", "_r12_normalize_output", "_witness_hash",
    "_run_attack_command", "_resolve_worktree_cwd", "_r12_attack_command_denied",
    "_r12_is_inspector_only", "_r12_is_execution_class", "_r12_targeted_open_blocks",
    "_r12_required_by_file", "_r12_changed_file_set", "_r12_validate_and_run_entry",
    "_r12_empty_set_rejection", "_pre_fix_anticipation_rejection",
    "_post_fix_attacks_rejection", "_r12_find_pre_fix_artifact",
    "_r12_required_commands_path", "_r12_required_commands_or_deny",
    "_r12_gates_shape_rejection", "_r12_anticipation_rejection",
    "_mutations_rejection", "_r12_new_test_surfaces", "_r12_previous_relay_row",
    "_exclusions_rejection",
}


def _r12_sentinel_line_range(source: str) -> tuple[int, int]:
    lines = source.splitlines()
    begin = next(i + 1 for i, l in enumerate(lines) if l.strip() == "# R12-BEGIN")
    end = next(i + 1 for i, l in enumerate(lines) if l.strip() == "# R12-END")
    return begin, end


def _r12_arm_id(source: str, pos: tuple[int, int]) -> str:
    """Fix round 5 Z10: a STABLE, human-readable identifier for a deny arm
    — `<enclosing top-level function name>: <the `if` line's own source
    text, stripped>` — used for the COMMITTED survivor list
    (`ci/scripts/r12_sweep_survivors.txt`), never the raw `(lineno,
    col_offset)` pair the sweep uses internally (which shifts on any
    unrelated edit earlier in the file and would make the committed list
    impossible to review or diff meaningfully)."""
    tree = ast.parse(source)
    lineno, col = pos
    enclosing = "?"
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            enclosing = node.name
            break
    lines = source.splitlines()
    line_text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else "?"
    return f"{enclosing}: {line_text}"


def _r12_deny_if_positions(source: str, begin: int, end: int) -> list[tuple[int, int]]:
    """`[(lineno, col_offset), ...]` of every distinct `If` node whose body
    contains a deny-shaped `Return` (a `Return` whose value is NOT the bare
    `None` constant) inside one of `_R12_SWEEP_FUNCS`, within the sentinel
    line range — de-duplicated, order-preserving. `ast.walk(node)` DOES
    descend into a NESTED FunctionDef (`_install_self_alarm`'s own
    `_handler`, e.g.) — there is no special-casing to exclude one; a deny
    arm inside a nested def is swept exactly like one at the enclosing
    function's own top level."""
    tree = ast.parse(source)
    parent_of: dict[ast.AST, ast.AST] = {}
    positions: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in _R12_SWEEP_FUNCS):
            continue
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parent_of[child] = parent
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Return):
                continue
            if not (begin <= sub.lineno <= end):
                continue
            val = sub.value
            if isinstance(val, ast.Constant) and val.value is None:
                continue  # the plain "accepted" return, never a deny
            cur: ast.AST | None = parent_of.get(sub)
            while cur is not None and not isinstance(cur, ast.If):
                cur = parent_of.get(cur)
            if isinstance(cur, ast.If):
                pos = (cur.lineno, cur.col_offset)
                if pos not in seen:
                    seen.add(pos)
                    positions.append(pos)
    return positions


def _r12_top_level_region_funcs(source: str, begin: int, end: int) -> set[str]:
    """The top-level `FunctionDef` NAME set inside the `[begin, end]` line
    range — module-level defs only (nested defs, e.g. `_install_self_
    alarm`'s own `_handler`, are excluded: they are swept as PART OF their
    enclosing top-level def, never listed separately)."""
    tree = ast.parse(source)
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and begin <= n.lineno <= end}


def fixture_r12sweepast_sweep_funcs_equals_the_sentinel_region() -> None:
    """esc-lead-gate-R12 fix round 5 Z6: `_R12_SWEEP_FUNCS` must equal —
    by AST, against the REAL file, never a hand-counted comment — the
    top-level `FunctionDef` set of the `# R12-BEGIN`/`# R12-END` sentinel
    region (nested defs excluded). RED by construction: inserts a new
    top-level def into a SYNTHETIC copy of the region and confirms the
    (deliberately unmodified) `_R12_SWEEP_FUNCS` no longer equals it —
    proving this fixture would actually catch a region def silently added
    without updating the set, not merely that the two currently agree by
    accident."""
    source = LEAD_GATE_LIB.read_text()
    begin, end = _r12_sentinel_line_range(source)
    real = _r12_top_level_region_funcs(source, begin, end)
    _assert(real == _R12_SWEEP_FUNCS, "R12sweepast",
            f"_R12_SWEEP_FUNCS must equal the sentinel region's own top-level FunctionDef set: "
            f"missing={real - _R12_SWEEP_FUNCS}, extra={_R12_SWEEP_FUNCS - real}")

    # The RED half: a synthetic copy of the region with ONE new top-level
    # def inserted must no longer equal the (real, unmodified) set.
    lines = source.splitlines()
    anchor = next(i for i, l in enumerate(lines) if l.strip() == "# R12-BEGIN")
    mutated_lines = lines[:anchor + 1] + [
        "def _r12_sweepast_canary_new_def():", "    return None", "",
    ] + lines[anchor + 1:]
    mutated_source = "\n".join(mutated_lines)
    m_begin, m_end = _r12_sentinel_line_range(mutated_source)
    mutated_real = _r12_top_level_region_funcs(mutated_source, m_begin, m_end)
    _assert(mutated_real != _R12_SWEEP_FUNCS, "R12sweepast",
            "inserting a new top-level def into the sentinel region must break the equality "
            "this fixture asserts — it did not, so this fixture cannot actually catch that case")
    _assert("_r12_sweepast_canary_new_def" in mutated_real, "R12sweepast setup",
            "the synthetic insertion did not land inside the mutated sentinel range")


class _R12NeuterIf(ast.NodeTransformer):
    def __init__(self, target: tuple[int, int]) -> None:
        self.target = target
        self.hit = False

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if (node.lineno, node.col_offset) == self.target:
            node.test = ast.copy_location(ast.Constant(value=False), node.test)
            self.hit = True
        return node


def _r12_mutate_at(source: str, target: tuple[int, int]) -> str:
    tree = ast.parse(source)
    transformer = _R12NeuterIf(target)
    mutated = transformer.visit(tree)
    ast.fix_missing_locations(mutated)
    _assert(transformer.hit, "R12-SWEEP setup", f"target If at {target} not found in a fresh parse")
    return ast.unparse(mutated)


def _r12_mutant_hooks_dir(mutated_source: str) -> Path:
    """A fresh temp dir carrying an unmodified copy of every real
    `.claude/hooks/*.sh` wrapper plus the ONE mutated `lead-gate-lib.py` —
    `lead-gate-pre.sh` resolves its sibling lib via its OWN dirname, so
    pointing `HOOKS_DIR` (module-global, monkey-patched for the duration of
    one mutant's fixture subset) at this directory is enough to exercise
    the mutated mechanism through the REAL wrapper scripts, unmodified."""
    d = tempfile.TemporaryDirectory(prefix="r12-mutant-hooks-")
    _TEMP_DIRS.append(d)
    p = Path(d.name)
    for sh in HOOKS_DIR.glob("*.sh"):
        (p / sh.name).write_text(sh.read_text())
        (p / sh.name).chmod(0o755)
    (p / "lead-gate-lib.py").write_text(mutated_source)
    return p


def _r12_run_fixture_subset_against(hooks_dir: Path, fixtures: list[tuple[str, object]]) -> list[str]:
    """Runs `fixtures` with `HOOKS_DIR` monkey-patched to `hooks_dir`,
    returning the names CREDITED as having killed this mutant.

    Fix round 5 Z10 (the HONEST credit rule): a death is credited ONLY
    when the fixture raises `Failure` (this harness's own `_assert`
    exception type) with a message that STARTS WITH the fixture's OWN
    registered `name` followed by `": "` — i.e. `_assert`'s `label`
    argument was that fixture's own name, the shape every fixture's
    TERMINAL assertion about its own expected ALLOW/DENY uses (`_assert(
    cond, "R12G1", detail)` raises `Failure(f"R12G1: {detail}")`).
    Two classes of "died" are explicitly EXCLUDED from credit, both of
    which the pre-fix sweep counted as a kill:
      1. A raw, non-`Failure` crash (an unrelated `TypeError`/`OSError`/
         etc. propagating from deep in a mutated call path) — before this
         fix, a single mutation that breaks an unrelated code path
         (`_r12_required_commands`'s own missing-file arm, neutered, used
         to raise `[Errno 2]` deep inside `.read_text()`) could crash
         DOZENS of fixtures that were never designed to exercise that arm
         at all, over-crediting it as "well covered" for the wrong
         reason.
      2. A `Failure` raised by a SETUP helper under a DIFFERENT label
         (e.g. `_assert(..., "R12G1 setup", ...)` or `_git`'s own "git
         fixture setup") — this fixture's OWN terminal assertion about
         the arm never even ran; crediting the kill to environment/setup
         breakage would be crediting the arm for a fixture that never
         actually reached it."""
    global HOOKS_DIR
    real_hooks_dir = HOOKS_DIR
    HOOKS_DIR = hooks_dir
    died: list[str] = []
    try:
        for name, fn in fixtures:
            try:
                fn()
            except Failure as exc:
                if str(exc).startswith(f"{name}: "):
                    died.append(name)
            except Exception:
                pass
    finally:
        HOOKS_DIR = real_hooks_dir
    return died


def run_r12_deny_coverage_sweep(fixtures: list[tuple[str, object]],
                                 source: str | None = None) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[str]]]:
    """`source` defaults to the REAL `lead-gate-lib.py` — overridable so the
    sweep meta-fixture (`R12sweepmeta`) can run this exact mechanism
    against a deliberately mutated COPY carrying one genuinely silent arm,
    proving the sweep's OWN detection logic fires, never merely that every
    arm CURRENTLY has a fixture."""
    if source is None:
        source = LEAD_GATE_LIB.read_text()
    begin, end = _r12_sentinel_line_range(source)
    positions = _r12_deny_if_positions(source, begin, end)
    # esc-lead-gate-R12 fix round 2 F4: R12D1 (`_r12_mod()`'s CACHED module
    # reference), R12alarm, R12timeout and R12alarmkill (all three load
    # `LEAD_GATE_LIB` — the real, un-mutated path constant — directly, by
    # construction, never `HOOKS_DIR`) can NEVER observe a mutant's
    # `HOOKS_DIR` monkey-patch; including them wastes every arm's budget
    # (R12alarmkill alone costs ~7s of real sleeps PER ARM) on fixtures
    # that can by definition never die, and would misleadingly VALIDATE a
    # mutant hooks dir that the rest of the subset never actually
    # exercised.
    _R12_MUTATION_BLIND = {"R12D1", "R12alarm", "R12timeout", "R12alarmkill", "R12sweepast"}
    r12_fixtures = [(n, f) for n, f in fixtures if n.startswith("R12") and n not in _R12_MUTATION_BLIND]
    per_arm: dict[tuple[int, int], list[str]] = {}
    for pos in positions:
        mutated = _r12_mutate_at(source, pos)
        hooks_dir = _r12_mutant_hooks_dir(mutated)
        per_arm[pos] = _r12_run_fixture_subset_against(hooks_dir, r12_fixtures)
    return positions, per_arm


def fixture_r12alarmkill_self_alarm_kills_inflight_attack_process_group() -> None:
    """esc-lead-gate-R12 fix round 2 advisory: the self-alarm's SIGALRM
    handler `killpg`s the CURRENTLY in-flight attack subprocess's own
    process group before `os._exit(2)` — without this, `os._exit` skips
    normal cleanup and the attack's own child (started in its own process
    group via `start_new_session=True`) is orphaned and keeps running
    after the hook process is gone. Spawns a throwaway subprocess that
    arms a 1s self-alarm then calls `_run_attack_command` on `sleep 3 &&
    touch <marker>` with a generous deadline (the ALARM fires first, never
    the command's own budget); after the subprocess dies, THIS fixture
    waits past the sleep's own 3s duration before checking — decisive: if
    the child had merely been orphaned rather than killed, it would have
    finished and created the marker well within that wait."""
    with tempfile.TemporaryDirectory(prefix="r12alarmkill-") as d:
        marker = str(Path(d) / "marker")
        proc = subprocess.run(
            ["python3", "-c",
             "import importlib.util, time, sys\n"
             f"spec = importlib.util.spec_from_file_location('m', {str(LEAD_GATE_LIB)!r})\n"
             "mod = importlib.util.module_from_spec(spec)\n"
             "spec.loader.exec_module(mod)\n"
             "mod._SELF_ALARM_S = 1\n"
             "mod._install_self_alarm()\n"
             f"mod._run_attack_command('sleep 3 && touch {marker}', {d!r}, time.monotonic() + 60)\n"],
            capture_output=True, text=True, timeout=10,
        )
        _assert(proc.returncode == 2, "R12alarmkill",
                f"the self-alarm must still fire and exit 2, got {proc.returncode}: {proc.stderr!r}")
        time.sleep(4.0)  # past the sleep's own 3s -- decisive, not a race
        _assert(not Path(marker).exists(), "R12alarmkill",
                "the marker file exists after waiting past the sleep's own duration -- the "
                "attack subprocess was orphaned and kept running instead of being killed")


def fixture_r12residual_marker_parsing() -> None:
    """esc-lead-gate-R12 fix round 2 F4: `_r12_residual_reason` recognizes
    a trailing `# R12-RESIDUAL: <reason>` comment on the EXACT line, and
    returns `None` (unmarked) for an ordinary line with no such comment —
    parsed precisely, never by a loose substring match that could
    accidentally swallow prose that merely mentions "residual". Cheap and
    pure (no subprocess, no mutation) — safe inside `--self-test`, unlike
    `R12sweepmeta` (which drives the real, slow mutation pipeline and
    belongs only in `--r12-sweep`)."""
    lines = [
        "    if some_condition:  # R12-RESIDUAL: dead code, unreachable by construction",
        "    if other_condition:",
        "    if third:  # a comment that mentions RESIDUAL but not the marker shape",
    ]
    _assert(_r12_residual_reason(lines, 1) == "dead code, unreachable by construction", "R12residual", lines[0])
    _assert(_r12_residual_reason(lines, 2) is None, "R12residual", "an unmarked line must return None")
    _assert(_r12_residual_reason(lines, 3) is None, "R12residual",
            "prose mentioning residual without the exact `# R12-RESIDUAL:` marker must not match")


def fixture_r12credit_unrelated_crash_not_credited() -> None:
    """fix round 5 Z10 (the crash-credit meta-fixture): a synthetic fixture
    whose ONLY failure is an unrelated crash (never this harness's own
    `_assert`/`Failure`) must NOT be credited by `_r12_run_fixture_subset_
    against` as having killed a mutant; a fixture whose failure IS its own
    terminal `_assert` (a `Failure` labeled with ITS OWN registered name)
    DOES get credited; a fixture that raises nothing survives (never
    credited either way). RED at daebd948: the pre-fix rule counted ANY
    exception, of any type, as a death — a mutation that broke something
    UNRELATED, deep in a shared code path, could crash dozens of fixtures
    never designed to exercise that arm and over-credit it as covered.
    Cheap and pure — no subprocess, no real mutation, safe inside
    `--self-test`."""
    def _crashes() -> None:
        raise RuntimeError("an unrelated crash, never this harness's own Failure")

    def _kills() -> None:
        _assert(False, "r12credit_synthetic_kills", "expected deny text never appeared")

    def _setup_failure_wrong_label() -> None:
        # A Failure IS raised, but under a DIFFERENT label than this
        # fixture's own registered name -- exactly the "setup helper"
        # shape (`_git`'s own "git fixture setup", or a "<name> setup"
        # helper) that must NOT be credited to the fixture's own name.
        _assert(False, "r12credit_synthetic_setup_failure setup", "unrelated setup breakage")

    def _survives() -> None:
        pass

    died = _r12_run_fixture_subset_against(HOOKS_DIR, [
        ("r12credit_synthetic_crash", _crashes),
        ("r12credit_synthetic_kills", _kills),
        ("r12credit_synthetic_setup_failure", _setup_failure_wrong_label),
        ("r12credit_synthetic_survives", _survives),
    ])
    _assert("r12credit_synthetic_crash" not in died, "R12credit",
            f"an unrelated crash must NOT be credited as a kill: {died}")
    _assert("r12credit_synthetic_kills" in died, "R12credit",
            f"a proper terminal _assert failure (labeled with its own name) must be credited: {died}")
    _assert("r12credit_synthetic_setup_failure" not in died, "R12credit",
            f"a Failure raised under a DIFFERENT label (a setup helper) must NOT be credited: {died}")
    _assert("r12credit_synthetic_survives" not in died, "R12credit",
            f"a fixture that raises nothing must not be credited either way: {died}")


def fixture_r12reqfile_real_file_shape() -> None:
    """fix round 5 Z4: `--self-test` really asserts shape against the REAL
    `ci/lead-gate-required-commands.txt` (never a fixture stand-in) --
    exists, names >=1 command line, every line carries a `# measured ~Xs`
    annotation, and every line's `python3 <script>`/`bash <script>`/
    `sh <script>` second token is a REAL, git-TRACKED path under
    REPO_ROOT."""
    path = REPO_ROOT / "ci" / "lead-gate-required-commands.txt"
    _assert(path.exists(), "R12reqfile", f"{path} must exist")
    tracked_out = _git(REPO_ROOT, "ls-files")
    tracked = set(tracked_out.splitlines())
    commands = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        _assert(re.search(r"#\s*measured\s*~", stripped) is not None, "R12reqfile",
                f"line {stripped!r} carries no `# measured ~Xs` annotation")
        command = stripped.split("  #", 1)[0].rstrip()
        _assert(bool(command), "R12reqfile", f"line {stripped!r} names no command after stripping the annotation")
        commands.append(command)
    _assert(len(commands) >= 1, "R12reqfile", f"{path} names no command line at all")
    for command in commands:
        tokens = command.split()
        script = tokens[1] if len(tokens) >= 2 and tokens[0] in ("python3", "bash", "sh") else None
        _assert(script is not None, "R12reqfile", f"{command!r} does not name a python3/bash/sh script")
        _assert(script in tracked, "R12reqfile", f"{script!r} named by {command!r} is not git-TRACKED at HEAD")


def fixture_r12reqmiss_missing_required_commands_file_denies() -> None:
    """fix round 5 Z4: a MISSING `ci/lead-gate-required-commands.txt`
    (removed after `_temp_repo`'s own baseline seed) is now a hard DENY at
    dispatch — never the old silent "no gate obligation" default."""
    unit = "feat/r12reqmiss"
    root = _temp_repo(unit)
    (root / "ci" / "lead-gate-required-commands.txt").unlink()
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}}, gates=None)
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12reqmiss", f"a missing required-commands file must deny, got {p.returncode}: {p.stderr}")
    _assert("does not exist" in p.stderr, "R12reqmiss", p.stderr)


def fixture_r12reqempty_allcomment_required_commands_file_denies() -> None:
    """fix round 5 Z4: an EMPTY/all-comment `ci/lead-gate-required-
    commands.txt` is likewise a hard DENY at dispatch."""
    unit = "feat/r12reqempty"
    root = _temp_repo(unit)
    (root / "ci" / "lead-gate-required-commands.txt").write_text("# nothing but comments\n")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}}, gates=None)
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12reqempty", f"an all-comment required-commands file must deny, got {p.returncode}: {p.stderr}")
    _assert("names no command line" in p.stderr, "R12reqempty", p.stderr)


def fixture_r12reqmiss2_relay_missing_required_commands_file_denies() -> None:
    """fix round 6 Z13 (Z4's other half, closing audit #4): R12reqmiss
    exercises READER 1 (the pre-fix dispatch); this fixture exercises
    READER 2 (the relay/re-audit path), which used to call the
    COLLAPSING `_r12_required_commands()` accessor (`[]` on a missing
    file) and fed that straight into `_r12_gates_shape_rejection`'s own
    `if not required_commands: return None` early-out — a relay with NO
    `gates` object at all was ALLOWED the instant the committed required-
    commands file was deleted, even though the IDENTICAL relay DENIES
    (R12G3) when the file is present. RED at 3273f51b by the executed
    probe: `p.returncode == 0` (ALLOW) with the file unlinked. GREEN after
    migrating reader 2 to `_r12_required_commands_or_deny()`, matching
    reader 1's own posture: a missing file denies EVERY relay, `gates` or
    not."""
    unit = "feat/r12reqmiss2"
    root = _temp_repo(unit)
    (root / "ci" / "lead-gate-required-commands.txt").unlink()
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "remove required-commands file")
    (root / "state.txt").write_text("BROKEN\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed state.txt = BROKEN/v1")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["state.txt:1"], ["state.txt:1"])
    cmd = "head -1 state.txt"
    pre_hash = _r12_hash(0, "BROKEN\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {"state.txt": {"command": cmd, "hash": pre_hash}})
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}},
                        override={"gates": None})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12reqmiss2",
            f"a relay must deny when the required-commands file is missing, got {p.returncode}: {p.stderr}")
    _assert("does not exist" in p.stderr, "R12reqmiss2", p.stderr)


def fixture_r12reqempty2_relay_allcomment_required_commands_file_denies() -> None:
    """fix round 6 Z13: the SAME reader-2 migration, the all-comment case.
    RED at 3273f51b by the executed probe: `p.returncode == 0` (ALLOW)
    with the file emptied to all-comment."""
    unit = "feat/r12reqempty2"
    root = _temp_repo(unit)
    (root / "ci" / "lead-gate-required-commands.txt").write_text("# nothing but comments\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "empty out the required-commands file")
    (root / "state.txt").write_text("BROKEN\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed state.txt = BROKEN/v1")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["state.txt:1"], ["state.txt:1"])
    cmd = "head -1 state.txt"
    pre_hash = _r12_hash(0, "BROKEN\n", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {"state.txt": {"command": cmd, "hash": pre_hash}})
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}},
                        override={"gates": None})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12reqempty2",
            f"a relay must deny when the required-commands file is all-comment, got {p.returncode}: {p.stderr}")
    _assert("names no command line" in p.stderr, "R12reqempty2", p.stderr)


def fixture_r12norm2_cargo_timing_normalized() -> None:
    """esc-lead-gate-R12 fix round 2 item 12: an UNMUTATED, execution-class
    command's own cargo/libtest-shaped summary line varies ONLY in its
    wall-clock timing between two runs — `_witness_hash` must still hash
    them IDENTICALLY (the property the differential's "post_hash !=
    pre_hash" check depends on: a hash inequality must mean something
    REALLY changed, never that cargo simply took a different number of
    milliseconds this run). A GENUINE behavioural change (a different pass
    count) still changes the hash — normalization strips ONLY the timing
    text, nothing else."""
    mod = _r12_mod()
    run1 = ("running 3 tests\n...\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 0.42s\n")
    run2 = ("running 3 tests\n...\ntest result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; "
            "0 filtered out; finished in 1.07s\n")
    run3_real_change = ("running 3 tests\n...\ntest result: ok. 2 passed; 1 failed; 0 ignored; "
                         "0 measured; 0 filtered out; finished in 0.42s\n")
    h1 = mod._witness_hash(0, run1, "")
    h2 = mod._witness_hash(0, run2, "")
    h3 = mod._witness_hash(0, run3_real_change, "")
    _assert(h1 == h2, "R12norm2",
            "two unmutated cargo-shaped runs differing ONLY in timing must hash identically")
    _assert(h1 != h3, "R12norm2",
            "a REAL behavioural change (a different pass/fail count) must still change the hash")


_R12_SWEEP_META_ANCHOR = 'human SWARM_GATE_TOUCHED review, and same-repo `on: pull_request`."""\n'
_R12_SWEEP_META_CANARY = (
    '    if command == "__r12_sweep_meta_canary__":\n'
    '        return "unreachable canary arm (R12sweepmeta)"  # noqa: R12-BEGIN sweep meta-fixture canary\n'
)


def fixture_r12sweepmeta_sweep_flags_a_genuinely_silent_arm() -> None:
    """esc-lead-gate-R12 fix round 2 acceptance fold: a meta-test on the
    SWEEP MECHANISM ITSELF, not on the current arm set — inserts ONE
    deliberately unreachable deny arm (`command ==
    "__r12_sweep_meta_canary__"`, a literal no fixture ever passes) into a
    throwaway COPY of the real lib's source, right inside
    `_r12_attack_command_denied` (already in `_R12_SWEEP_FUNCS`), and runs
    the REAL `run_r12_deny_coverage_sweep` against that copy. Proves the
    sweep correctly reports this canary as a SURVIVOR (no dying fixture)
    — never that every arm CURRENTLY in the tree happens to have one,
    which a bug in the sweep's own detection/mutation/re-run logic could
    satisfy vacuously (e.g. a mutation that silently no-ops, or a fixture
    subset that never actually runs)."""
    source = LEAD_GATE_LIB.read_text()
    _assert(source.count(_R12_SWEEP_META_ANCHOR) == 1, "R12sweepmeta",
            "the sweep meta-fixture's anchor text no longer appears exactly once in the real "
            "lib — update the anchor to match the current source")
    mutated = source.replace(_R12_SWEEP_META_ANCHOR, _R12_SWEEP_META_ANCHOR + _R12_SWEEP_META_CANARY, 1)
    _assert(mutated != source, "R12sweepmeta", "the canary insertion did not change the source")
    canary_line = None
    for i, line in enumerate(mutated.splitlines(), start=1):
        if '"__r12_sweep_meta_canary__"' in line:
            canary_line = i
            break
    _assert(canary_line is not None, "R12sweepmeta", "could not locate the injected canary line")
    # A cheap, representative subset (never the full ~50+ R12* fixture
    # list — this meta-fixture is about the MECHANISM, not re-proving
    # every real arm; using a handful keeps this fast). Never R12D1/
    # R12alarm/R12timeout — mutation-blind by construction (F4), which
    # `run_r12_deny_coverage_sweep` already excludes from ITS OWN subset,
    # but listing a mutation-blind name here would prove nothing anyway.
    subset = [(n, f) for n, f in FIXTURES if n in ("R12P9", "R12P13")]
    _assert(len(subset) == 2, "R12sweepmeta setup", f"expected 2 fixtures in the subset, got {len(subset)}")
    positions, per_arm = run_r12_deny_coverage_sweep(subset, source=mutated)
    canary_positions = [pos for pos in positions if pos[0] == canary_line]
    _assert(len(canary_positions) == 1, "R12sweepmeta",
            f"the injected canary arm was not detected as a distinct position: {positions}")
    canary_died = per_arm[canary_positions[0]]
    _assert(canary_died == [], "R12sweepmeta",
            f"the sweep must report the genuinely unreachable canary arm as a SURVIVOR "
            f"(no dying fixture), but it reported: {canary_died}")
    # Robustness: a broken mutation/re-run pipeline (a mutant hooks dir
    # that never actually gets exercised, say) would ALSO make the canary
    # look like a survivor -- for the WRONG reason. Confirm at least one
    # NON-canary, real arm DOES die under this same subset, proving the
    # pipeline actually distinguishes mutants at all.
    non_canary_deaths = [died for pos, died in per_arm.items() if pos[0] != canary_line and died]
    _assert(non_canary_deaths, "R12sweepmeta",
            "no non-canary arm died under this subset -- the mutation/re-run pipeline itself "
            "may be broken (everything would look like a survivor for the wrong reason)")


# ==========================================================================
# esc-lead-gate-R12 fix round 3, item 8 (the fold-9 replacement: 8a/8b/8c).
# ==========================================================================

def fixture_r12g1_pre_fix_missing_gates_denies() -> None:
    """item 8a: a required-commands file is committed, but the
    anticipation artifact carries no `gates` object at all -- Reader 1
    denies on SHAPE, never touching the value of any `rc`."""
    unit = "feat/r12g1"
    root = _temp_repo(unit)
    _write_required_commands_file(root, ["python3 ci/probe_r12g1.py"])
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    }, gates=None)
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12G1", f"a missing `gates` object must deny, got {p.returncode}: {p.stderr}")
    _assert("no `gates` object" in p.stderr, "R12G1", p.stderr)


def fixture_r12g2_pre_fix_gates_shape_only_allows_nonzero_rc() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a shape-complete `gates` object must not be
    denied merely because `rc` is non-zero pre-fix). The RED half is
    R12G1/R12G3-4/R12G6-8 (each denies a specific shape defect this
    fixture's OWN artifact does NOT carry). item 8a: Reader 1 never judges
    the VALUE of `rc` -- a `gates` entry recorded `rc=1` (as if the
    required command is currently failing on the BROKEN pre-fix tip)
    still allows, as long as the SHAPE (every committed line, verbatim,
    each an object with an integer `rc`) is complete."""
    unit = "feat/r12g2"
    root = _temp_repo(unit)
    _write_required_commands_file(root, ["python3 ci/probe_r12g2.py"])
    (root / "probe_r12g2.sh").write_text("#!/bin/sh\nprintf ok\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "add probe_r12g2.sh")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    cmd = "bash probe_r12g2.sh"
    h = _r12_hash(0, "ok", "")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": cmd, "hash": h},
    }, gates={"python3 ci/probe_r12g2.py": {"rc": 1}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 0, "R12G2",
            f"a shape-complete gates object with a non-zero rc must still allow pre-fix, got {p.returncode}: {p.stderr}")


def fixture_r12g3_relay_missing_gates_denies() -> None:
    """item 8a: reader 2 (the relay) denies when a required-commands file
    is committed but the relay carries no `gates` object at all."""
    root, row, cmd, pre_hash = _r12_gates_post_setup("feat/r12g3", ["python3 ci/scripts/probe.py"])
    _write_anticipation_exact(root, "feat/r12g3", row["head_sha"], {"state.txt": {"command": cmd, "hash": pre_hash}})
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}},
                        override={"gates": None})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12g3"}}, root)
    _assert(p.returncode == 2, "R12G3", f"relay missing `gates` must deny, got {p.returncode}: {p.stderr}")
    _assert("no `gates` object" in p.stderr, "R12G3", p.stderr)


def fixture_r12g4_relay_gates_nonzero_rc_denies() -> None:
    """item 8a: reader 2 REQUIRES `rc == 0` for every committed line at
    fix_head -- a relay `gates` entry recording a non-zero rc denies,
    even though it has the right shape."""
    root, row, cmd, pre_hash = _r12_gates_post_setup("feat/r12g4", ["python3 ci/scripts/probe.py"])
    _write_anticipation_exact(root, "feat/r12g4", row["head_sha"], {"state.txt": {"command": cmd, "hash": pre_hash}})
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}},
                        override={"gates": {"python3 ci/scripts/probe.py": {"rc": 1}}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12g4"}}, root)
    _assert(p.returncode == 2, "R12G4", f"a non-zero post-fix gate rc must deny, got {p.returncode}: {p.stderr}")
    _assert("recorded rc=1" in p.stderr, "R12G4", p.stderr)


def fixture_r12g5_relay_gates_rc_zero_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a shape-complete, fully green relay `gates`
    object must allow). The RED half is R12G1/R12G3-4/R12G6-8. item 8a:
    the satisfiable case -- every committed line present with `rc == 0`
    allows (the whole relay's other obligations being otherwise
    satisfied)."""
    root, row, cmd, pre_hash = _r12_gates_post_setup("feat/r12g5", ["python3 ci/scripts/probe.py"])
    _write_anticipation_exact(root, "feat/r12g5", row["head_sha"], {"state.txt": {"command": cmd, "hash": pre_hash}})
    (root / "state.txt").write_text("FIXED\nv1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: state.txt line 1 = FIXED")
    fix_head = _git(root, "rev-parse", "HEAD")
    post_hash = _r12_hash(0, "FIXED\n", "")
    _write_relay_exact(root, row, sites={"state.txt:1": "fixed"}, probe=["c.py:9", "state.txt"],
                        fix_head=fix_head,
                        attacks_post={"state.txt": {"command": cmd, "hash": post_hash}},
                        override={"gates": {"python3 ci/scripts/probe.py": {"rc": 0}}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": "re-audit unit: feat/r12g5"}}, root)
    _assert(p.returncode == 0, "R12G5", f"a fully green gates object must allow, got {p.returncode}: {p.stderr}")


def fixture_r12g6_pre_fix_gates_partial_completeness_denies() -> None:
    """item 8a: TWO committed lines, but the artifact's `gates` names only
    ONE -- partial completeness (distinct from R12G1's totally-missing
    `gates` object)."""
    unit = "feat/r12g6"
    root = _temp_repo(unit)
    _write_required_commands_file(root, ["python3 ci/probe_a.py", "python3 ci/probe_b.py"])
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    }, gates={"python3 ci/probe_a.py": {"rc": 0}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12G6", f"partial gates completeness must deny, got {p.returncode}: {p.stderr}")
    _assert("omits" in p.stderr and "committed command" in p.stderr, "R12G6", p.stderr)


def fixture_r12g7_pre_fix_gates_entry_not_object_denies() -> None:
    """item 8a: a `gates` value that is not an object at all."""
    unit = "feat/r12g7"
    root = _temp_repo(unit)
    _write_required_commands_file(root, ["python3 ci/probe_a.py"])
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    }, gates={"python3 ci/probe_a.py": "not-an-object"})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12G7", f"a non-object gates entry must deny, got {p.returncode}: {p.stderr}")
    _assert("is not an object" in p.stderr, "R12G7", p.stderr)


def fixture_r12g8_pre_fix_gates_no_integer_rc_denies() -> None:
    """item 8a: a `gates` entry with no integer `rc` field."""
    unit = "feat/r12g8"
    root = _temp_repo(unit)
    _write_required_commands_file(root, ["python3 ci/probe_a.py"])
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {
        "a.py": {"command": a["command"], "hash": a["hash"]},
    }, gates={"python3 ci/probe_a.py": {"note": "forgot rc"}})
    p = _r12_dispatch(root, unit)
    _assert(p.returncode == 2, "R12G8", f"a gates entry with no integer rc must deny, got {p.returncode}: {p.stderr}")
    _assert("carries no integer `rc`" in p.stderr, "R12G8", p.stderr)


def _r12_mutations_setup(unit: str):
    """A block row whose own `finding_locations` names `bar.py`, and a fix
    that adds a REAL new definition (`def helper():`) in that SAME file --
    the DATA-armed condition item 8b's `_mutations_rejection` requires."""
    root = _temp_repo(unit)
    (root / "bar.py").write_text("x = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "seed bar.py")
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["bar.py:1"], ["bar.py:1"])
    _write_anticipation_exact(root, unit, row["head_sha"], {"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, "x = 1\n", "")}})
    (root / "bar.py").write_text("x = 1\n\n\ndef helper():\n    return x\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: bar.py adds helper()")
    fix_head = _git(root, "rev-parse", "HEAD")
    return root, row, fix_head


def fixture_r12m8b1_missing_mutations_denies() -> None:
    """item 8b: armed by the DATA -- the fix adds a new definition
    (`helper()`) in `bar.py`, a file the BLOCK's own finding_locations
    also names -- and the relay carries no `mutations` array at all."""
    unit = "feat/r12m8b1"
    root, row, fix_head = _r12_mutations_setup(unit)
    post_hash = _r12_hash(0, "x = 1", "")
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12M8b1", f"a missing `mutations` array must deny, got {p.returncode}: {p.stderr}")
    _assert("no `mutations` array" in p.stderr, "R12M8b1", p.stderr)


def fixture_r12m8b2_too_many_mutations_denies() -> None:
    """item 8b: `mutations` is capped at K<=3, a LABELED sample."""
    unit = "feat/r12m8b2"
    root, row, fix_head = _r12_mutations_setup(unit)
    rows = [{"site": f"bar.py:{i}", "command": "true", "rc_before": 0, "rc_after": 1,
             "marker_after": "test result: FAILED"} for i in range(4)]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": rows})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12M8b2", f"more than 3 mutation rows must deny, got {p.returncode}: {p.stderr}")
    _assert("K<=3" in p.stderr, "R12M8b2", p.stderr)


def fixture_r12m8b3_accepted_mutation_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a correctly-shaped ACCEPTED mutation row must
    allow). The RED half is R12M8b1-2/R12M8b5-12. item 8b: the
    satisfiable ACCEPT case -- `rc_before == 0`, `rc_after != 0`, and
    `marker_after` names a committed TEST-failure marker."""
    unit = "feat/r12m8b3"
    root, row, fix_head = _r12_mutations_setup(unit)
    rows = [{"site": "bar.py:4", "command": "true", "rc_before": 0, "rc_after": 1,
             "marker_after": "test result: FAILED. 0 passed; 1 failed"}]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": rows})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 0, "R12M8b3", f"an accepted mutation row must allow, got {p.returncode}: {p.stderr}")


def fixture_r12m8b4_uncovered_mutation_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a correctly-shaped `uncovered` disposition must
    allow). The RED half is R12M8b1-2/R12M8b5-12. item 8b: an explicit
    `uncovered` reason (R11's own disposition precedent) satisfies the
    obligation without an accepted mutation."""
    unit = "feat/r12m8b4"
    root, row, fix_head = _r12_mutations_setup(unit)
    rows = [{"site": "bar.py:4", "command": "true", "uncovered": "no test harness reaches this call site directly"}]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": rows})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 0, "R12M8b4", f"an uncovered mutation row must allow, got {p.returncode}: {p.stderr}")


def fixture_r12m8b5_rejected_mutation_denies() -> None:
    """item 8b: a mutation row that neither satisfies the ACCEPT rule
    (here: `rc_after == 0`, no actual break) NOR carries an `uncovered`
    reason denies."""
    unit = "feat/r12m8b5"
    root, row, fix_head = _r12_mutations_setup(unit)
    rows = [{"site": "bar.py:4", "command": "true", "rc_before": 0, "rc_after": 0,
             "marker_after": "test result: ok"}]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": rows})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12M8b5", f"a rejected mutation row with no uncovered reason must deny, got {p.returncode}: {p.stderr}")
    _assert("does not satisfy the ACCEPT rule" in p.stderr, "R12M8b5", p.stderr)


def _r12m8b_shape_case(label: str, row_value: object, expected_text: str) -> None:
    """Shared shape-defect case for the M8b6-M8b12 fixtures below -- ONE
    malformed `mutations[0]` value each, asserting the SPECIFIC deny text
    `_mutations_rejection` names for that determinant."""
    unit = f"feat/{label.lower()}"
    root, row, fix_head = _r12_mutations_setup(unit)
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": [row_value]})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, label, f"a malformed mutation row ({label}) must deny, got {p.returncode}: {p.stderr}")
    _assert(expected_text in p.stderr, label, p.stderr)


def fixture_r12m8b6_mutation_entry_not_object_denies() -> None:
    """item 8b: `mutations[0]` is not an object at all."""
    _r12m8b_shape_case("R12M8b6", "not-a-dict", "is not an object")


def fixture_r12m8b7_mutation_missing_site_denies() -> None:
    """item 8b: `mutations[0]` carries no `site`."""
    _r12m8b_shape_case("R12M8b7", {"command": "true", "rc_before": 0, "rc_after": 1,
                                    "marker_after": "test result: FAILED"}, "carries no `site`")


def fixture_r12m8b8_mutation_missing_command_denies() -> None:
    """item 8b: `mutations[0]` carries no `command`."""
    _r12m8b_shape_case("R12M8b8", {"site": "bar.py:4", "rc_before": 0, "rc_after": 1,
                                    "marker_after": "test result: FAILED"}, "carries no `command`")


def fixture_r12m8b9_mutation_empty_uncovered_denies() -> None:
    """item 8b: `mutations[0]`'s `uncovered` field is present but empty."""
    _r12m8b_shape_case("R12M8b9", {"site": "bar.py:4", "command": "true", "uncovered": "   "},
                        "`uncovered` is present but empty")


def fixture_r12m8b10_mutation_missing_rc_before_denies() -> None:
    """item 8b: `mutations[0]` carries no integer `rc_before`."""
    _r12m8b_shape_case("R12M8b10", {"site": "bar.py:4", "command": "true", "rc_after": 1,
                                     "marker_after": "test result: FAILED"}, "carries no integer `rc_before`")


def fixture_r12m8b11_mutation_missing_rc_after_denies() -> None:
    """item 8b: `mutations[0]` carries no integer `rc_after`."""
    _r12m8b_shape_case("R12M8b11", {"site": "bar.py:4", "command": "true", "rc_before": 0,
                                     "marker_after": "test result: FAILED"}, "carries no integer `rc_after`")


def fixture_r12m8b12_mutation_missing_marker_after_denies() -> None:
    """item 8b: `mutations[0]` carries no `marker_after`."""
    _r12m8b_shape_case("R12M8b12", {"site": "bar.py:4", "command": "true", "rc_before": 0, "rc_after": 1},
                        "carries no `marker_after`")


def fixture_r12m8b13_identical_uncovered_reasons_deny() -> None:
    """fix round 5 Z11 (audit advisory 8): three `mutations` rows (the K<=3
    cap's own limit), each carrying the IDENTICAL (normalized) `uncovered`
    reason -- a templated disposition satisfying the row-shape obligation
    three times over with ONE real examination, never denied before this
    fix (R11's own distinctness precedent was never reused here)."""
    unit = "feat/r12m8b13"
    root, row, fix_head = _r12_mutations_setup(unit)
    same = "no test harness reaches this call site directly"
    rows = [{"site": f"bar.py:{i}", "command": "true", "uncovered": same} for i in (4, 5, 6)]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": rows})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12M8b13", f"three identical uncovered reasons must deny, got {p.returncode}: {p.stderr}")
    _assert("IDENTICAL `uncovered` reason" in p.stderr, "R12M8b13", p.stderr)


def _r12_exclusions_setup(unit: str):
    """A fix that adds a REAL new TEST definition (`tests/test_x.py`'s
    `def test_thing():`) -- the DATA-armed condition item 8c's
    `_exclusions_rejection` requires (via `_r12_new_test_surfaces`)."""
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}})
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_x.py").write_text("def test_thing():\n    assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add tests/test_x.py")
    fix_head = _git(root, "rev-parse", "HEAD")
    return root, row, fix_head


def fixture_r12x1_missing_exclusions_denies() -> None:
    """item 8c: the fix's own diff adds a new test definition
    (`test_thing`), and the relay carries no `exclusions` object at all."""
    unit = "feat/r12x1"
    root, row, fix_head = _r12_exclusions_setup(unit)
    a = _auto_r12_attack("a.py")
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head,
                        attacks_post={"a.py": {"command": a["command"], "hash": a["hash"]}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12X1", f"missing `exclusions` must deny, got {p.returncode}: {p.stderr}")
    _assert("no `exclusions` object" in p.stderr, "R12X1", p.stderr)


def fixture_r12x2_exclusions_present_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a correctly-shaped, non-empty, distinct
    `exclusions` entry must allow). The RED half is
    R12X1/R12X3-4/R12X5-6. item 8c: the satisfiable case -- a non-empty
    exclusion named for the new test definition."""
    unit = "feat/r12x2"
    root, row, fix_head = _r12_exclusions_setup(unit)
    a = _auto_r12_attack("a.py")
    key = [k for k, v in _parse_new_surfaces_of(root, row["head_sha"], fix_head).items() if "test_thing" in v][0]
    test_content = (root / "tests" / "test_x.py").read_text()
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_x.py": {"command": "cat tests/test_x.py", "hash": _r12_hash(0, test_content, "")},
                        },
                        override={"exclusions": {key: "does not cover an assertion failure, only that the function runs"}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 0, "R12X2", f"a real exclusion must allow, got {p.returncode}: {p.stderr}")


def fixture_r12x3_empty_exclusion_denies() -> None:
    """item 8c: an `exclusions` entry present but empty denies."""
    unit = "feat/r12x3"
    root, row, fix_head = _r12_exclusions_setup(unit)
    a = _auto_r12_attack("a.py")
    key = [k for k, v in _parse_new_surfaces_of(root, row["head_sha"], fix_head).items() if "test_thing" in v][0]
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head,
                        attacks_post={"a.py": {"command": a["command"], "hash": a["hash"]}},
                        override={"exclusions": {key: "   "}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12X3", f"an empty exclusion must deny, got {p.returncode}: {p.stderr}")
    _assert("is empty" in p.stderr, "R12X3", p.stderr)


def _r12_exclusions_two_setup(unit: str):
    """The SAME shape as `_r12_exclusions_setup`, but the fix adds TWO new
    test definitions (`test_thing`/`test_other`) -- needed for the
    partial-completeness (R12X4) and within-relay-duplicate (R12X5)
    cases, neither of which a single new test definition can exercise."""
    root = _temp_repo(unit)
    row = _write_block_row(root, unit, "a1", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    a = _auto_r12_attack("a.py")
    _write_anticipation_exact(root, unit, row["head_sha"], {"a.py": {"command": a["command"], "hash": a["hash"]}})
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_x.py").write_text("def test_thing():\n    assert True\n\n\ndef test_other():\n    assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add tests/test_x.py with two tests")
    fix_head = _git(root, "rev-parse", "HEAD")
    return root, row, fix_head


def fixture_r12x4_exclusions_partial_completeness_denies() -> None:
    """item 8c: TWO new test definitions, but `exclusions` names only
    ONE -- distinct from R12X1's totally-missing `exclusions` object."""
    unit = "feat/r12x4"
    root, row, fix_head = _r12_exclusions_two_setup(unit)
    a = _auto_r12_attack("a.py")
    surfaces = _parse_new_surfaces_of(root, row["head_sha"], fix_head)
    thing_key = [k for k, v in surfaces.items() if "test_thing" in v][0]
    test_content = (root / "tests" / "test_x.py").read_text()
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_x.py": {"command": "cat tests/test_x.py", "hash": _r12_hash(0, test_content, "")},
                        },
                        override={"exclusions": {thing_key: "does not cover an assertion failure"}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12X4", f"partial exclusions completeness must deny, got {p.returncode}: {p.stderr}")
    _assert("omits" in p.stderr and "new test definition" in p.stderr, "R12X4", p.stderr)


def fixture_r12x5_exclusions_duplicate_within_relay_denies() -> None:
    """item 8c: two new test definitions, both given the IDENTICAL
    (normalized) exclusion text -- a templated disposition, never a
    per-test examination."""
    unit = "feat/r12x5"
    root, row, fix_head = _r12_exclusions_two_setup(unit)
    a = _auto_r12_attack("a.py")
    surfaces = _parse_new_surfaces_of(root, row["head_sha"], fix_head)
    thing_key = [k for k, v in surfaces.items() if "test_thing" in v][0]
    other_key = [k for k, v in surfaces.items() if "test_other" in v][0]
    test_content = (root / "tests" / "test_x.py").read_text()
    same_text = "does not cover an assertion failure"
    _write_relay_exact(root, row, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_x.py": {"command": "cat tests/test_x.py", "hash": _r12_hash(0, test_content, "")},
                        },
                        override={"exclusions": {thing_key: same_text, other_key: same_text}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 2, "R12X5", f"a duplicated within-relay exclusion must deny, got {p.returncode}: {p.stderr}")
    _assert("IDENTICAL exclusion" in p.stderr, "R12X5", p.stderr)


def fixture_r12x6_exclusions_duplicate_across_previous_relay_denies() -> None:
    """item 8c: the unit's own SECOND adversarial-audit round adds a new
    test definition and names the IDENTICAL (normalized) exclusion text
    the FIRST round's own accepted relay already used -- a templated
    exclusion carried across rounds, never a per-round examination."""
    unit = "feat/r12x6"
    root, row1, fix_head1 = _r12_exclusions_setup(unit)
    a = _auto_r12_attack("a.py")
    surfaces1 = _parse_new_surfaces_of(root, row1["head_sha"], fix_head1)
    key1 = [k for k, v in surfaces1.items() if "test_thing" in v][0]
    test_content1 = (root / "tests" / "test_x.py").read_text()
    same_text = "does not cover an assertion failure, only that the function runs"
    _write_relay_exact(root, row1, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head1,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_x.py": {"command": "cat tests/test_x.py", "hash": _r12_hash(0, test_content1, "")},
                        },
                        override={"exclusions": {key1: same_text}})
    p1 = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p1.returncode == 0, "R12X6 setup", f"the FIRST round's relay must itself allow, got {p1.returncode}: {p1.stderr}")

    row2 = _write_block_row(root, unit, "a2", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    (root / "tests" / "test_y.py").write_text("def test_second():\n    assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add tests/test_y.py")
    fix_head2 = _git(root, "rev-parse", "HEAD")
    surfaces2 = _parse_new_surfaces_of(root, row2["head_sha"], fix_head2)
    key2 = [k for k, v in surfaces2.items() if "test_second" in v][0]
    test_content2 = (root / "tests" / "test_y.py").read_text()
    _write_relay_exact(root, row2, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_y.py"],
                        fix_head=fix_head2,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_y.py": {"command": "cat tests/test_y.py", "hash": _r12_hash(0, test_content2, "")},
                        },
                        override={"exclusions": {key2: same_text}})
    p2 = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p2.returncode == 2, "R12X6", f"a cross-relay duplicated exclusion must deny, got {p2.returncode}: {p2.stderr}")
    _assert("PREVIOUS relay" in p2.stderr, "R12X6", p2.stderr)


def fixture_r12x7_missing_previous_relay_artifact_denies() -> None:
    """fix round 5 Z9: the SAME two-round shape as R12X6, but the FIRST
    round's own relay artifact is `rm`'d off disk BEFORE the second round
    dispatches -- the ledger (`docs/rigor`-style row history) still
    records that a previous round happened, so the cross-round
    distinctness check cannot silently treat "no artifact to compare
    against" as "nothing to compare, therefore fine": it must DENY,
    naming the missing witness (the SAME shape F1's missing-pre-fix-
    artifact arm takes at lib:2122)."""
    unit = "feat/r12x7"
    root, row1, fix_head1 = _r12_exclusions_setup(unit)
    a = _auto_r12_attack("a.py")
    surfaces1 = _parse_new_surfaces_of(root, row1["head_sha"], fix_head1)
    key1 = [k for k, v in surfaces1.items() if "test_thing" in v][0]
    test_content1 = (root / "tests" / "test_x.py").read_text()
    _write_relay_exact(root, row1, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_x.py"],
                        fix_head=fix_head1,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_x.py": {"command": "cat tests/test_x.py", "hash": _r12_hash(0, test_content1, "")},
                        },
                        override={"exclusions": {key1: "does not cover an assertion failure, only that the function runs"}})
    p1 = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p1.returncode == 0, "R12X7 setup", f"the FIRST round's relay must itself allow, got {p1.returncode}: {p1.stderr}")

    # The FIRST round's own relay artifact is removed -- the ledger row
    # (`_write_block_row`'s own committed history) still names it, but the
    # witness itself is gone.
    first_relay_path = _relay_path_exact(root, row1)
    _assert(first_relay_path.exists(), "R12X7 setup", "the first round's relay artifact must exist before removal")
    first_relay_path.unlink()

    row2 = _write_block_row(root, unit, "a2", "adversarial-audit", ["a.py:1"], ["a.py:1"])
    (root / "tests" / "test_y.py").write_text("def test_second():\n    assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fix: add tests/test_y.py")
    fix_head2 = _git(root, "rev-parse", "HEAD")
    surfaces2 = _parse_new_surfaces_of(root, row2["head_sha"], fix_head2)
    key2 = [k for k, v in surfaces2.items() if "test_second" in v][0]
    test_content2 = (root / "tests" / "test_y.py").read_text()
    _write_relay_exact(root, row2, sites={"a.py:1": "fixed"}, probe=["c.py:9", "tests/test_y.py"],
                        fix_head=fix_head2,
                        attacks_post={
                            "a.py": {"command": a["command"], "hash": a["hash"]},
                            "tests/test_y.py": {"command": "cat tests/test_y.py", "hash": _r12_hash(0, test_content2, "")},
                        },
                        override={"exclusions": {key2: "a distinct exclusion text, never seen before"}})
    p2 = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p2.returncode == 2, "R12X7", f"a missing previous-relay witness must deny, got {p2.returncode}: {p2.stderr}")
    _assert("no on-disk artifact" in p2.stderr, "R12X7", p2.stderr)


def fixture_r12x8_export_emits_no_attestation_row_or_file() -> None:
    """Round-6 stop rule fired (audit #5 BLOCK at da30f0b2 on Z12's export
    half): fix round 5 Z8's export half is REVERTED entirely.
    `--export-anticipation` writes EXACTLY ONE stream — stdout,
    `lead-anticipation` rows only. The lead's `mutations`/`exclusions`
    attestations stay HOOK-ATTESTED ONLY: they remain in the relay
    artifact readers 1 and 2 already read, and are never exported, never
    written to a committed file, never counted in this command's own
    stderr summary. A relay carrying non-empty `mutations`/`exclusions`
    (the exact fixture that used to produce a `lead-relay-attestation`
    row) must now produce NEITHER a stdout row of that kind NOR a
    `docs/rigor/<slug>.attestation.jsonl` file at all."""
    unit = "feat/r12x8"
    root, row, fix_head = _r12_mutations_setup(unit)
    mutation_rows = [{"site": "bar.py:4", "command": "true", "rc_before": 0, "rc_after": 1,
                       "marker_after": "test result: FAILED. 0 passed; 1 failed"}]
    _write_relay_exact(root, row, sites={"bar.py:1": "fixed"}, probe=["c.py:9", "bar.py"],
                        fix_head=fix_head,
                        attacks_post={"bar.py": {"command": "cat bar.py", "hash": _r12_hash(0, (root / "bar.py").read_text(), "")}},
                        override={"mutations": mutation_rows, "exclusions": {"tests/test_z.py::test_new": "does not cover X"}})
    p = _run("lead-gate-pre.sh", {"tool_name": "Agent", "tool_input": {
        "subagent_type": "adversarial-audit", "prompt": f"re-audit unit: {unit}"}}, root)
    _assert(p.returncode == 0, "R12X8 setup", f"the relay must itself allow, got {p.returncode}: {p.stderr}")

    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(root)
    proc = subprocess.run(
        [sys.executable, str(LEAD_GATE_LIB), "--export-anticipation", _slug(unit)],
        capture_output=True, text=True, env=env, timeout=10,
    )
    _assert(proc.returncode == 0, "R12X8", f"--export-anticipation must exit 0: {proc.stderr}")
    exported_rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    _assert(all(r.get("agent_type") == "lead-anticipation" for r in exported_rows), "R12X8",
            f"the anticipation stdout stream must carry NO lead-relay-attestation row: {exported_rows}")
    attestation_path = root / "docs" / "rigor" / f"{_slug(unit)}.attestation.jsonl"
    _assert(not attestation_path.exists(), "R12X8",
            f"the exporter must write NO attestation file at all (Z8's export half is reverted): "
            f"{attestation_path}")
    _assert("attestation" not in proc.stderr, "R12X8",
            f"the exporter's own stderr summary must not mention attestation rows: {proc.stderr!r}")


def _parse_new_surfaces_of(root: Path, base_sha: str, head_sha: str) -> dict[str, str]:
    """Test-harness helper: the SAME `_parse_new_surfaces` the hook itself
    uses, applied to a real `-U0` diff in `root` -- used only to derive
    the EXACT key a fixture must name in `exclusions`, never duplicating
    the hook's own detection logic."""
    mod = _r12_mod()
    diff_out = _git(root, "diff", "-U0", "--end-of-options", base_sha, head_sha)
    return mod._parse_new_surfaces(diff_out)


FIXTURES = [
    ("R12G1", fixture_r12g1_pre_fix_missing_gates_denies),
    ("R12G2", fixture_r12g2_pre_fix_gates_shape_only_allows_nonzero_rc),
    ("R12G3", fixture_r12g3_relay_missing_gates_denies),
    ("R12G4", fixture_r12g4_relay_gates_nonzero_rc_denies),
    ("R12G5", fixture_r12g5_relay_gates_rc_zero_allows),
    ("R12G6", fixture_r12g6_pre_fix_gates_partial_completeness_denies),
    ("R12G7", fixture_r12g7_pre_fix_gates_entry_not_object_denies),
    ("R12G8", fixture_r12g8_pre_fix_gates_no_integer_rc_denies),
    ("R12M8b1", fixture_r12m8b1_missing_mutations_denies),
    ("R12M8b2", fixture_r12m8b2_too_many_mutations_denies),
    ("R12M8b3", fixture_r12m8b3_accepted_mutation_allows),
    ("R12M8b4", fixture_r12m8b4_uncovered_mutation_allows),
    ("R12M8b5", fixture_r12m8b5_rejected_mutation_denies),
    ("R12M8b6", fixture_r12m8b6_mutation_entry_not_object_denies),
    ("R12M8b7", fixture_r12m8b7_mutation_missing_site_denies),
    ("R12M8b8", fixture_r12m8b8_mutation_missing_command_denies),
    ("R12M8b9", fixture_r12m8b9_mutation_empty_uncovered_denies),
    ("R12M8b10", fixture_r12m8b10_mutation_missing_rc_before_denies),
    ("R12M8b11", fixture_r12m8b11_mutation_missing_rc_after_denies),
    ("R12M8b12", fixture_r12m8b12_mutation_missing_marker_after_denies),
    ("R12M8b13", fixture_r12m8b13_identical_uncovered_reasons_deny),
    ("R12X1", fixture_r12x1_missing_exclusions_denies),
    ("R12X2", fixture_r12x2_exclusions_present_allows),
    ("R12X3", fixture_r12x3_empty_exclusion_denies),
    ("R12X4", fixture_r12x4_exclusions_partial_completeness_denies),
    ("R12X5", fixture_r12x5_exclusions_duplicate_within_relay_denies),
    ("R12X6", fixture_r12x6_exclusions_duplicate_across_previous_relay_denies),
    ("R12X7", fixture_r12x7_missing_previous_relay_artifact_denies),
    ("R12X8", fixture_r12x8_export_emits_no_attestation_row_or_file),
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
    ("UC1", fixture_uc1_missing_claim_denied),
    ("UC2", fixture_uc2_tested_claim_with_matching_hash_allows),
    ("UC3", fixture_uc3_hash_mismatch_denied),
    ("UC4", fixture_uc4_write_verb_denied),
    ("UC5", fixture_uc5_uncovered_with_reason_allows),
    ("UC6", fixture_uc6_duplicate_uncovered_reason_denied),
    ("UC7", fixture_uc7_no_claim_shaped_line_is_a_noop),
    ("R12P1", fixture_r12p1_no_unit_line_denies_for_implementer_types),
    ("R12P1b", fixture_r12p1b_no_unit_line_allows_for_extra_gated_types),
    ("R12P1c", fixture_r12p1c_extra_gated_type_armed_when_unit_named),
    ("R12P2", fixture_r12p2_reader1_not_armed_no_open_block),
    ("R12P2b", fixture_r12p2b_unresolvable_branch_denies),
    ("R12P3", fixture_r12p3_missing_artifact_denies),
    ("R12P4", fixture_r12p4_missing_key_denies),
    ("R12P5", fixture_r12p5_tip_moved_since_artifact_recorded_denies),
    ("R12P5b", fixture_r12p5b_dirty_worktree_denies_naming_paths),
    ("R12P6", fixture_r12p6_hash_mismatch_denies),
    ("R12P7", fixture_r12p7_vacuous_missing_script_denies),
    ("R12P8", fixture_r12p8_templated_reused_pair_denies),
    ("R12P9", fixture_r12p9_denylisted_command_denies),
    ("R12P10", fixture_r12p10_complete_artifact_allows),
    ("R12P10b", fixture_r12p10b_inspector_only_artifact_denies),
    ("R12P10c", fixture_r12p10c_printf_only_artifact_denies),
    ("R12P11", fixture_r12p11_two_open_blocks_different_shas_still_satisfiable),
    ("R12P11b", fixture_r12p11b_union_still_requires_the_older_blocks_own_keys),
    ("R12P12", fixture_r12p12_computed_cwd_not_project_dir),
    ("R12P13", fixture_r12p13_bash_nonexistent_path_denies_via_dispatch),
    ("R12P14", fixture_r12p14_git_subcommand_denied_via_dispatch),
    ("R12P15", fixture_r12p15_find_delete_denied_via_dispatch),
    ("R12P16", fixture_r12p16_missing_command_field_denies),
    ("R12P17", fixture_r12p17_invalid_hash_shape_denies),
    ("R12P18", fixture_r12p18_no_linked_worktree_denies),
    ("R12P19", fixture_r12p19_execution_class_direct_program_name_allows),
    ("R12P20", fixture_r12p20_non_dict_attack_entry_denies),
    ("R12E1", fixture_r12e1_empty_derived_set_fewer_than_two_keys_denies),
    ("R12E2", fixture_r12e2_empty_derived_set_key_outside_changed_files_denies),
    ("R12E3", fixture_r12e3_empty_derived_set_all_inspector_denies),
    ("R12E4", fixture_r12e4_empty_derived_set_two_valid_execution_class_allows),
    ("R12E5", fixture_r12e5_empty_derived_set_unresolvable_main_denies),
    ("R12E6", fixture_r12e6_empty_derived_set_malformed_entry_denies),
    ("R12E7", fixture_r12e7_empty_derived_set_printf_only_denies),
    ("R12E8", fixture_r12e8_empty_derived_set_reused_pair_denies),
    ("R12F1", fixture_r12f1_pre_fix_sha_mismatch_denies),
    ("R12F2", fixture_r12f2_unit_branch_field_mismatch_denies),
    ("R12F3", fixture_r12f3_missing_residual_risk_denies),
    ("R12F4", fixture_r12f4_non_dict_json_artifact_denies),
    ("R12F5", fixture_r12f5_attacks_field_missing_denies),
    ("R12R2f", fixture_r12r2f_partial_attacks_post_coverage_denies),
    ("R12R2g", fixture_r12r2g_attacks_post_entry_not_object_denies),
    ("R12R2h", fixture_r12r2h_attacks_post_command_differs_from_pre_fix_denies),
    ("R12R2i", fixture_r12r2i_attacks_post_hash_mismatch_denies),
    ("R12R2j", fixture_r12r2j_fix_changed_file_with_no_covering_key_denies),
    ("R12witness", fixture_r12witness_stderr_distinguishes_missing_from_real_failure),
    ("R12reduce", fixture_r12reduce_two_lines_same_file_one_required_entry),
    ("R12unparse", fixture_r12unparse_prose_key_maps_to_itself_still_required),
    ("R12D1", fixture_r12d1_relaxed_denylist_still_denies_bashc_and_pipe_sh),
    ("R12F1a", fixture_r12f1a_reader2_finds_artifact_by_covers_not_block_head_sha),
    ("R12F1b", fixture_r12f1b_deleted_artifact_denies_not_allows),
    ("R12R2a", fixture_r12r2a_missing_attacks_post_denies),
    ("R12R2b", fixture_r12r2b_no_differential_denies),
    ("R12R2c", fixture_r12r2c_real_differential_allows),
    ("R12R2d", fixture_r12r2d_sh_hunk_widens_by_file),
    ("R12R2e", fixture_r12r2e_yml_hunk_widens_by_file),
    ("R12M1", fixture_r12m1_open_question_without_attacks_post_denies),
    ("R12norm", fixture_r12norm_no_rm_in_any_deny_text),
    ("R12phase", fixture_r12phase_slow_attack_does_not_exhaust_next_git_phase),
    ("R12residual", fixture_r12residual_marker_parsing),
    ("R12credit", fixture_r12credit_unrelated_crash_not_credited),
    ("R12reqfile", fixture_r12reqfile_real_file_shape),
    ("R12reqmiss", fixture_r12reqmiss_missing_required_commands_file_denies),
    ("R12reqempty", fixture_r12reqempty_allcomment_required_commands_file_denies),
    ("R12reqmiss2", fixture_r12reqmiss2_relay_missing_required_commands_file_denies),
    ("R12reqempty2", fixture_r12reqempty2_relay_allcomment_required_commands_file_denies),
    ("R12sweepast", fixture_r12sweepast_sweep_funcs_equals_the_sentinel_region),
    ("R12norm2", fixture_r12norm2_cargo_timing_normalized),
    ("R12alarmkill", fixture_r12alarmkill_self_alarm_kills_inflight_attack_process_group),
    # R12sweepmeta is deliberately NOT here — like the sweep itself (see
    # `r12_sweep_main`'s own docstring), it re-runs the mutation/re-run
    # pipeline (measured ~60s) and belongs in `--r12-sweep`, never in the
    # per-invocation `--self-test` every unit's pressure-tester/oracle
    # round already re-runs.
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
    ("R12timeout", fixture_r12timeout_settings_pins_above_self_bound),
    ("R12alarm", fixture_r12alarm_self_bound_denies),
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


_R12_RESIDUAL_RE = re.compile(r"#\s*R12-RESIDUAL:\s*(.+?)\s*$")


def _r12_residual_reason(source_lines: list[str], lineno: int) -> str | None:
    """esc-lead-gate-R12 fix round 2 F4: a survivor arm carrying a
    committed `# R12-RESIDUAL: <reason>` trailing comment on its OWN `if`
    line is REPORTED, never FAILED — the sweep's honest-residual escape
    hatch for an arm this harness genuinely cannot exercise (a git-failure
    branch, dead code, an out-of-scope arm), distinct from a silently
    unmarked gap."""
    if not (1 <= lineno <= len(source_lines)):
        return None
    m = _R12_RESIDUAL_RE.search(source_lines[lineno - 1])
    return m.group(1) if m else None


def _r12_load_survivors() -> set[str]:
    """The committed survivor list (`ci/scripts/r12_sweep_survivors.txt`)
    — one arm ID per line, `#`-comment/blank lines skipped. `set()` when
    the file does not exist (the bootstrap state before this PR commits
    it for the first time)."""
    if not R12_SWEEP_SURVIVORS_PATH.exists():
        return set()
    out: set[str] = set()
    for line in R12_SWEEP_SURVIVORS_PATH.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def _real_git(cwd: Path, *args: str) -> tuple[bool, str]:
    """A non-raising git subprocess wrapper for REAL repo operations
    (fetching/reading `origin/main`) — distinct from this file's own
    `_git()` fixture helper, which `_assert`s success (correct for
    building a throwaway fixture repo, wrong for a ratchet that must
    report a git failure as its OWN FAIL, never crash the check)."""
    try:
        proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr)


def check_r12_survivors_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    """fix round 5 Z10: the SAME shrink-only ratchet shape as `check_rigor_
    record.py`'s `check_allowlist_only_shrinks`/`check_r12_grandfather_
    only_shrinks`, applied to `R12_SWEEP_SURVIVORS_PATH` — the committed
    set of KNOWN, currently-unfixed silent deny arms can never GROW
    autonomously on a swarm branch; only shrink (a fixture added, or a
    `# R12-RESIDUAL` marker attached, removes an arm from the sweep's own
    unmarked-survivor set, and the human then deletes its line here).
    BOOTSTRAP arm: `origin/main` carrying no such file yet (this PR is
    the one introducing it) establishes the baseline instead of failing."""
    ok, _ = _real_git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("check-lead-gate[r12-survivors-only-shrinks]: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _real_git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("check-lead-gate[r12-survivors-only-shrinks]: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = _r12_load_survivors()
    rel = R12_SWEEP_SURVIVORS_PATH.relative_to(cwd).as_posix()
    ok, base_text = _real_git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"check-lead-gate[r12-survivors-only-shrinks]: OK (bootstrap) — origin/main has no "
              f"{rel} yet; this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("check-lead-gate[r12-survivors-only-shrinks]: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\ncheck-lead-gate[r12-survivors-only-shrinks]: this branch adds a NEW accepted "
              "survivor. The list may only shrink — a genuinely new exemption is a human-reviewed "
              "decision, made on main directly, never an autonomous addition on a swarm branch.",
              file=sys.stderr)
        return 1
    print(f"check-lead-gate[r12-survivors-only-shrinks]: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


def r12_sweep_main() -> int:
    """esc-lead-gate-R12 M5' — its OWN `swarm.yml` step, separate from
    `--self-test` (mutating and re-running R12 fixtures per deny arm is too
    slow to fold into the per-invocation self-test every unit's pressure-
    tester/oracle round already re-runs). Also runs `R12sweepmeta` (a
    meta-test on the sweep mechanism itself — same reason it is excluded
    from `--self-test`) BEFORE trusting the real sweep's own result.

    Fix round 5 Z10: the unmarked-survivor arm is REPORT-ONLY — it never
    fails this step (a step that mutates and re-runs the ENTIRE R12
    fixture set per arm is too slow/flaky to be the hard gate; a genuinely
    NEW unmarked survivor is instead surfaced loudly here for a human to
    triage, and the only thing that can ever make one PERMANENTLY
    accepted is `--check-r12-survivors-only-shrinks` refusing to let the
    committed list grow). Only a MECHANISM failure (R12sweepmeta itself,
    or the sweep pipeline crashing) still returns non-zero."""
    print("check-lead-gate[R12-SWEEP]: running R12sweepmeta (mechanism self-check) first...")
    try:
        fixture_r12sweepmeta_sweep_flags_a_genuinely_silent_arm()
    except Failure as e:
        print(f"check-lead-gate[R12-SWEEP]: FAIL — R12sweepmeta itself failed: {e}", file=sys.stderr)
        return 1
    print("check-lead-gate[R12-SWEEP]: R12sweepmeta OK — the mechanism correctly flags a "
          "genuinely silent arm; proceeding to the real sweep.")

    source = LEAD_GATE_LIB.read_text()
    source_lines = source.splitlines()
    start = time.monotonic()
    positions, per_arm = run_r12_deny_coverage_sweep(FIXTURES, source=source)
    elapsed = time.monotonic() - start
    all_survivors = [pos for pos, died in per_arm.items() if not died]
    residual: dict[tuple[int, int], str] = {}
    unmarked_survivors: list[tuple[int, int]] = []
    for pos in all_survivors:
        reason = _r12_residual_reason(source_lines, pos[0])
        if reason:
            residual[pos] = reason
        else:
            unmarked_survivors.append(pos)
    killed = len(positions) - len(all_survivors)
    unmarked_ids = {_r12_arm_id(source, pos) for pos in unmarked_survivors}
    committed = _r12_load_survivors()
    new_unmarked = sorted(unmarked_ids - committed)
    stale_committed = sorted(committed - unmarked_ids)
    print(f"check-lead-gate[R12-SWEEP]: {len(positions)} deny arm(s) swept in {elapsed:.2f}s "
          f"({killed} killed by >=1 fixture, {len(residual)} marked residual, "
          f"{len(unmarked_survivors)} unmarked survivor(s), {len(committed)} committed to "
          f"{R12_SWEEP_SURVIVORS_PATH.name})")
    for pos in sorted(per_arm):
        died = per_arm[pos]
        if died:
            status = f"dies via {', '.join(died)}"
        elif pos in residual:
            status = f"R12-RESIDUAL: {residual[pos]}"
        else:
            aid = _r12_arm_id(source, pos)
            status = ("SURVIVES (committed)" if aid in committed else "SURVIVES (NEW, NOT COMMITTED)")
        print(f"  arm@line{pos[0]}: {status}")
    if new_unmarked:
        print(f"check-lead-gate[R12-SWEEP]: {len(new_unmarked)} NEW unmarked survivor(s) not yet "
              f"on {R12_SWEEP_SURVIVORS_PATH.name} — REPORT-ONLY (fix round 5 Z10), never a step "
              "failure: mark a genuine residual with `# R12-RESIDUAL: <reason>`, add a fixture "
              "that kills it, or have a human commit it to the survivor list on main directly.")
        for aid in new_unmarked:
            print(f"    NEW: {aid}")
    if stale_committed:
        print(f"check-lead-gate[R12-SWEEP]: {len(stale_committed)} committed survivor(s) no longer "
              "reproduce as unmarked — shrink the list to match (never grows autonomously, but a "
              "human should shrink it once an arm is actually fixed).")
        for aid in stale_committed:
            print(f"    STALE: {aid}")
    print(f"check-lead-gate[R12-SWEEP]: OK (report-only on survivors) — {killed} arm(s) killed, "
          f"{len(residual)} residual, {len(unmarked_survivors)} unmarked ({elapsed:.2f}s)")
    return 0


def main() -> int:
    if "--r12-sweep" in sys.argv[1:]:
        return r12_sweep_main()
    if "--check-r12-survivors-only-shrinks" in sys.argv[1:]:
        return check_r12_survivors_only_shrinks()
    if "--self-test" in sys.argv[1:]:
        return self_test()
    print("check_lead_gate.py: usage: --self-test | --r12-sweep | "
          "--check-r12-survivors-only-shrinks", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
