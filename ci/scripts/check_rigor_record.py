#!/usr/bin/env python3
"""check_rigor_record.py — R7b (esc-lead-gate-R7, v2/REFINE): a required
check that a lead cannot satisfy by under-reporting, because it never reads
anything the lead authors to decide WHETHER it applies.

Two design rounds killed the earlier shapes of this idea (see
`docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md` for the
full history, citations, and the measurements below). The one property that
survived every round: **arm on the DIFF's own shape, never on a count or a
flag the constrained party writes about itself.**

THE RULE. If `base...HEAD` (a three-dot range, never two-dot — see "Known,
undetectable residual" below) touches any path under `crates/**`, `ci/**`,
or `.github/workflows/**`, this check is ARMED for the unit on this branch
and requires, from COMMITTED files only:

  1. A rigor record at `docs/rigor/<unit_slug>.jsonl` — the hook's own row
     schema (`lead-gate-lib.py --export <slug>`), never hand-typed — exists
     and parses.
  2. That record carries at least one row with `agent_type ==
     "pressure-tester"` — the design pass `.claude/agents/lead.md:41`
     already requires before any non-local fix, now visible in a committed
     diff a human reviews at merge, not merely asserted in a transcript.
  3. This SAME diff (`base...HEAD`) adds or modifies at least one file
     under `docs/plans/*/proposals/**` or `docs/rigor/contracts/**` — the
     committed mechanism contract itself.
  4. That contract passes the four ANTI-VACUITY cost-floor checks below.
     Nothing mechanical judges a contract's QUALITY — this repo states
     that plainly rather than implying otherwise.

`unit_slug` is `slugify(<head branch name>)`, the SAME `slugify()`
`lead-gate-lib.py` already uses (loaded dynamically from
`.claude/hooks/lead-gate-lib.py` — never reimplemented, so the two never
drift) — the PR's own head ref, read from `GITHUB_HEAD_REF` in Actions or
`git rev-parse --abbrev-ref HEAD` otherwise. This is the SAME identity a
verifier's `unit_branch` field already resolves to everywhere else in this
system; a round of adversarial replay over 473 real hook decisions found
gated verifier re-dispatches bind to it at ~98% (cited in the proposal
doc), the most reliable identity this system has.

ARMED BY THE DIFF, SATISFIED BY DISCLOSURE, JUDGED BY THE HUMAN.
TAMPER-EVIDENT, NEVER TAMPER-PROOF — never described any other way.
`.jammi/gate-state/` (the source the rigor record is exported FROM)
carries no `permissions.deny` entry and the lead already writes into it by
design — a hand-appended row is possible and undetectable at the hook
layer (see `README.md`'s mechanical-vs-visible-only list, item 5). What
this check adds is not tamper-PREVENTION: it is that the record becomes a
COMMITTED file in a reviewed diff, so a human
merging the PR can see "this unit touched a mechanism; here is its design
pass; here is its round history" — or see that one of those three is
silently missing, which a gitignored state directory could never show
them at all.

ANCESTRY IS ADVISORY, NEVER A FAILURE (check 3, historical numbering).
Every recorded `head_sha` in the rigor record is checked for whether it
resolves and is an ancestor of HEAD, using the exact git shape
`_relay_rejection` already uses in `lead-gate-lib.py` — but ONLY reported,
never failed on. Measured, twice, independently: this session's own
documented workflow (amend an implementer's commit to add session
trailers) orphans every head_sha a verifier recorded before the amend —
`git merge-base --is-ancestor <pre-amend sha> HEAD` exits 1 on the
honest path, and in a transport clone (what `actions/checkout` produces)
the object is ABSENT entirely. `_relay_rejection`'s identical shape only
works because it runs in a local repository whose object store still
holds the orphan; a CI checkout has no such luxury.

KNOWN, STRUCTURAL NO-OP SHAPES (never an author-declared marker/trailer/
commit-message convention — every such predicate is a one-line opt-out):
  - not a `pull_request` event (a push to `main` with no PR context) —
    `GITHUB_EVENT_NAME` unset or not `pull_request`.
  - the PR's base is not the default branch — `swarm.yml`'s own
    `on: pull_request: branches: [main]` trigger already guarantees this
    whenever the workflow runs at all; checked again here defensively.
  - dependabot — `GITHUB_ACTOR == "dependabot[bot]"`.
  - a docs-only/typo diff — implied structurally: a diff touching none of
    `crates/**`/`ci/**`/`.github/workflows/**` never arms in the first
    place; no special case is needed.
  - a release-shaped diff — every changed file under `crates/**` is one of
    `Cargo.toml`/`Cargo.lock` and no `ci/**`/`.github/workflows/**` file is
    touched (a version-bump-only PR). Stated as a heuristic, not a proof:
    a release PR that ALSO edits a gate script is not exempted by this
    rule, and should not be.
  - a purely human-authored PR — none of the commits in `base..HEAD`
    carries this repo's own `Co-Authored-By: Claude` trailer convention.
    A single swarm-touched commit in the range is enough to arm; the
    swarm's own rigor process does not bind a human contributor who never
    touched the branch.
  - a revert — the PR's HEAD commit subject starts with `Revert "`, git's
    own `git revert` convention.

KNOWN, UNDETECTABLE RESIDUAL, STATED HONESTLY: a retargeted stacked branch
whose three-dot range still carries a parent unit's own commits (the base
this PR's history actually diverged from is stale relative to the PR's
declared `base_ref`) is NOT detectable from the diff alone — three-dot
ranges are the standard, correct GitHub PR semantics, and there is no
mechanical way to distinguish "this diff correctly represents the PR's own
work" from "this branch needs a rebase" without knowing the intended base.
Pin three-dot ranges explicitly (done); do not pretend this residual is
closed.

Modes:
  python3 ci/scripts/check_rigor_record.py                       # the check
  python3 ci/scripts/check_rigor_record.py --check-allowlist-only-shrinks
  python3 ci/scripts/check_rigor_record.py --check-r12-grandfather-only-shrinks
  python3 ci/scripts/check_rigor_record.py --check-required-commands-only-shrinks
  python3 ci/scripts/check_rigor_record.py --self-test
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAD_GATE_LIB = REPO_ROOT / ".claude" / "hooks" / "lead-gate-lib.py"
RIGOR_DIR = REPO_ROOT / "docs" / "rigor"
ALLOWLIST_PATH = REPO_ROOT / "ci" / "scripts" / "rigor_record_allowlist.txt"
# esc-lead-gate-R12 (M3'): in-flight units whose second-round BLOCK predates
# fix round 1's anticipation mechanism land without a
# `docs/rigor/<slug>.anticipation.jsonl` record. This is the SAME
# shrink-only-ratchet SHAPE as `ALLOWLIST_PATH`/`check_allowlist_only_
# shrinks` above (modelled on it directly, per swarm.yml's own two-step
# pairing) — a SEPARATE file and a SEPARATE check, never folded into the
# general allowlist, because the two exemptions arm on different diffs
# (this one is read only by `check_anticipation_witnesses`, never by
# `run_check`'s own top-level arming).
R12_GRANDFATHER_PATH = REPO_ROOT / "ci" / "scripts" / "rigor_record_r12_grandfather.txt"
# esc-lead-gate-R12 fix round 5 Z4: the OPPOSITE polarity from the two
# exemption lists above — this file names REQUIRED gate commands, so
# GROWING it (adding a line) tightens the swarm and ADDING/keeping every
# existing line is fine; REMOVING a line (a shrink in the count of
# required commands) WEAKENS item 8a and is exactly what this ratchet
# catches and fails, never allows silently.
REQUIRED_COMMANDS_PATH = REPO_ROOT / "ci" / "lead-gate-required-commands.txt"

ARMING_GLOBS = ("crates/*", "crates/**", "ci/*", "ci/**", ".github/workflows/*", ".github/workflows/**")
CONTRACT_GLOBS = ("docs/plans/*/proposals/*", "docs/plans/*/proposals/**", "docs/rigor/contracts/*", "docs/rigor/contracts/**")
# A file-extension-bearing path token followed by `:<line>[-<line>]` — the
# same grammar family `_probe_path`/`_PROBE_LINESPEC_RE` in lead-gate-lib.py
# already uses for citation-checkable evidence, reused rather than
# reinvented (never a length/word-count rule — Goodhart, and it is
# prose-reading by another name).
PATH_LINE_RE = re.compile(r"`?([A-Za-z0-9_./-]+\.[A-Za-z0-9]+):(\d+)(?:-(\d+))?`?")


class Result:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.failures


def _lib_module():
    spec = importlib.util.spec_from_file_location("lead_gate_lib_rigor", LEAD_GATE_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _git(cwd: Path, *args: str) -> tuple[bool, str]:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr).strip()


def _display_path(p: Path) -> str:
    """`p` relative to REPO_ROOT when it is under it (the real invocation);
    otherwise the path as-is (a self-test fixture's own throwaway tree,
    which is never under REPO_ROOT)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _matches_arming_glob(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in ARMING_GLOBS)


def _matches_contract_glob(path: str) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in CONTRACT_GLOBS)


def compute_diff_context(cwd: Path, base_ref: str, head_ref: str = "HEAD") -> tuple[bool, list[str], str]:
    """Fetches `base_ref` from `origin` and returns `(ok, changed_paths,
    range_spec)` for `origin/<base_ref>...<head_ref>` — a THREE-DOT range,
    never two-dot (the range this whole mechanism is pinned to)."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", base_ref)
    if not ok:
        return False, [], ""
    ok, _ = _git(cwd, "rev-parse", "--verify", f"origin/{base_ref}")
    if not ok:
        return False, [], ""
    range_spec = f"origin/{base_ref}...{head_ref}"
    ok, out = _git(cwd, "diff", "--name-only", range_spec)
    if not ok:
        return False, [], range_spec
    return True, [p for p in out.splitlines() if p.strip()], range_spec


def commits_in_range(cwd: Path, range_spec: str) -> list[str]:
    ok, out = _git(cwd, "log", "--format=%H", range_spec.replace("...", ".."))
    return [c for c in out.splitlines() if c.strip()] if ok else []


def is_human_authored(cwd: Path, range_spec: str) -> bool:
    """True iff NO commit in the range carries this repo's own swarm
    co-authorship trailer — a single swarm-touched commit is enough to
    arm; a purely human PR that never touched the branch is not bound by
    the swarm's own rigor process."""
    shas = commits_in_range(cwd, range_spec)
    if not shas:
        return True
    for sha in shas:
        ok, msg = _git(cwd, "log", "-1", "--format=%B", sha)
        if ok and re.search(r"Co-Authored-By:\s*Claude", msg, re.IGNORECASE):
            return False
    return True


def is_revert(cwd: Path, head_ref: str = "HEAD") -> bool:
    ok, subject = _git(cwd, "log", "-1", "--format=%s", head_ref)
    return ok and subject.startswith('Revert "')


def is_release_shaped(changed: list[str]) -> bool:
    crate_touches = [p for p in changed if _matches_arming_glob(p) and p.startswith("crates/")]
    other_mechanism_touches = [
        p for p in changed
        if _matches_arming_glob(p) and not p.startswith("crates/")
    ]
    if other_mechanism_touches:
        return False
    if not crate_touches:
        return False
    return all(Path(p).name in ("Cargo.toml", "Cargo.lock") for p in crate_touches)


def _no_op_reason(cwd: Path, changed: list[str], range_spec: str) -> str | None:
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event and event != "pull_request":
        return f"not a pull_request event ({event!r})"
    actor = os.environ.get("GITHUB_ACTOR", "")
    if actor == "dependabot[bot]":
        return "dependabot PR"
    if is_revert(cwd):
        return "revert PR (HEAD commit subject starts with Revert \")"
    if is_release_shaped(changed):
        return "release-shaped diff (only Cargo.toml/Cargo.lock under crates/**, no ci/**/.github/workflows/** touch)"
    if is_human_authored(cwd, range_spec):
        return "no commit in range carries the swarm's own Co-Authored-By: Claude trailer"
    return None


# --- Cost-floor (anti-vacuity) checks -----------------------------------

def _path_lines_at_head(cwd: Path, path: str) -> int | None:
    ok, out = _git(cwd, "show", f"HEAD:{path}")
    if not ok:
        return None
    return len(out.splitlines())


def check_path_line_citations(cwd: Path, contract_path: str, text: str, result: Result) -> None:
    seen = set()
    for m in PATH_LINE_RE.finditer(text):
        cited_path, line1, line2 = m.group(1), int(m.group(2)), m.group(3)
        key = (cited_path, line1, line2)
        if key in seen:
            continue
        seen.add(key)
        n = _path_lines_at_head(cwd, cited_path)
        if n is None:
            result.fail(f"{contract_path}: cites {cited_path}:{line1}"
                        f"{'-' + line2 if line2 else ''} but {cited_path} does not exist at HEAD")
            continue
        top = int(line2) if line2 else line1
        if top > n:
            result.fail(f"{contract_path}: cites {cited_path}:{top} but {cited_path} has only {n} line(s) at HEAD")


def check_introducing_commit_ancestor(cwd: Path, contract_path: str, result: Result) -> None:
    ok, out = _git(cwd, "log", "--follow", "--diff-filter=A", "--format=%H", "--", contract_path)
    if not ok or not out.strip():
        result.warn(f"{contract_path}: could not find an introducing (add) commit via --follow")
        return
    introducing = out.strip().splitlines()[-1]
    ok, _ = _git(cwd, "merge-base", "--is-ancestor", introducing, "HEAD")
    if not ok:
        result.fail(f"{contract_path}: introducing commit {introducing} is not an ancestor of HEAD")


def _normalize_for_dedup(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def check_not_near_identical(cwd: Path, contract_path: str, text: str, all_contracts: list[str], result: Result) -> None:
    my_hash = hashlib.sha256(_normalize_for_dedup(text).encode("utf-8")).hexdigest()
    for other in all_contracts:
        if other == contract_path:
            continue
        ok, other_text = _git(cwd, "show", f"HEAD:{other}")
        if not ok:
            continue
        other_hash = hashlib.sha256(_normalize_for_dedup(other_text).encode("utf-8")).hexdigest()
        if other_hash == my_hash:
            result.fail(f"{contract_path}: near-identical (normalized-hash match) to {other}"
                        " — a copy-paste contract is not a design pass")


def _r12_grandfathered_slugs() -> set[str]:
    if not R12_GRANDFATHER_PATH.exists():
        return set()
    return {s.strip() for s in R12_GRANDFATHER_PATH.read_text().splitlines()
            if s.strip() and not s.strip().startswith("#")}


def _r12_second_round_block_rows(mod, rows: list[dict]) -> list[dict]:
    """Every row in the EXPORTED unit record whose `agent_type` is one of
    the hook's own `VERIFIER_SECOND_ROUND_TYPES` and whose `verdict` is
    BLOCK-equivalent (`is_open`) — the SAME closed-world membership the
    hook itself uses for reader 1, loaded from the real module, never
    re-derived here."""
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("agent_type") not in mod.VERIFIER_SECOND_ROUND_TYPES:
            continue
        verdict = r.get("verdict")
        if isinstance(verdict, str) and mod.is_open(verdict):
            out.append(r)
    return out


def check_anticipation_witnesses(cwd: Path, unit_slug: str, rows: list[dict], result: Result) -> None:
    """esc-lead-gate-R12 READER 3 (fix round 1, M3'): a committed-record
    SHAPE check, HARD-FAILED on shape alone. ARMED whenever the exported
    `docs/rigor/<unit_slug>.jsonl` carries ANY open second-round BLOCK row
    (`_r12_second_round_block_rows`) and `unit_slug` is NOT on the
    shrink-only `R12_GRANDFATHER_PATH` list — never on a ts cutover (M3'
    drops that; the grandfather list is the ONLY exemption, and it may only
    shrink, enforced by `check_r12_grandfather_only_shrinks`). When armed:
    `docs/rigor/<unit_slug>.anticipation.jsonl` (exported via
    `lead-gate-lib.py --export-anticipation`) must exist, every row must
    parse, and the UNION of every second-round BLOCK row's own
    `finding_locations` ∪ `class_enumeration`, reduced to files
    (`mod._key_to_file`), must be covered by SOME row's `attacks` keys —
    every command passing the hook's OWN denylist
    (`mod._r12_attack_command_denied`, loaded from the real
    `lead-gate-lib.py`, never reimplemented — `bash <path>` admitted only
    when TRACKED at the row's own `pre_fix_sha`). Re-EXECUTION against a
    REAL, independently checked-out `pre_fix_sha` is ADVISORY ONLY —
    reported per attack as reproduced / mismatched / not_re_executed, NEVER
    a failure (measured: BSD/GNU userland divergence between the lead's
    machine and CI's, and path divergence, both change a witness hash
    without the mechanism under attack having changed at all)."""
    mod = _lib_module()
    block_rows = _r12_second_round_block_rows(mod, rows)
    if not block_rows:
        return  # nothing to anticipate -- no open second-round BLOCK in this record

    path = f"docs/rigor/{unit_slug}.anticipation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    has_file = ok and bool(text.strip())

    if unit_slug in _r12_grandfathered_slugs():
        if not has_file:
            result.warn(f"{unit_slug!r} is on the shrink-only R12 grandfather list "
                        f"({_display_path(R12_GRANDFATHER_PATH)}) — no {path} carried; not required")
            return
    elif not has_file:
        result.fail(
            f"docs/rigor/{unit_slug}.jsonl carries {len(block_rows)} open second-round BLOCK "
            f"row(s) but no {path} — export one with `python3 .claude/hooks/lead-gate-lib.py "
            f"--export-anticipation {unit_slug} > {path}` and commit it (esc-lead-gate-R12 "
            "READER 3); an in-flight unit whose BLOCK predates fix round 1 is exempted only via "
            f"{_display_path(R12_GRANDFATHER_PATH)} (shrink-only, human-added)")
        return

    if not has_file:
        return

    required_files: set[str] = set()
    for row in block_rows:
        locs = {s for s in (row.get("finding_locations") or []) if isinstance(s, str)}
        enum = {s for s in (row.get("class_enumeration") or []) if isinstance(s, str)}
        for key in locs | enum:
            required_files.add(mod._key_to_file(key))

    art_rows: list[tuple[int, dict]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            result.fail(f"{path}:{i + 1}: not valid JSON ({exc})")
            continue
        if not isinstance(parsed, dict):
            result.fail(f"{path}:{i + 1}: not a JSON object")
            continue
        art_rows.append((i + 1, parsed))

    if not art_rows:
        result.fail(f"{path}: carries no parseable anticipation row")
        return

    covered_files: set[str] = set()
    for lineno, row in art_rows:
        attacks = row.get("attacks")
        pre_fix_sha = row.get("pre_fix_sha")
        if not isinstance(attacks, dict) or not attacks:
            result.fail(f"{path}:{lineno}: no non-empty `attacks` object")
            continue
        if not isinstance(pre_fix_sha, str) or not pre_fix_sha:
            result.fail(f"{path}:{lineno}: no `pre_fix_sha`")
        sha_resolvable = False
        if isinstance(pre_fix_sha, str) and pre_fix_sha:
            ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{pre_fix_sha}^{{commit}}")
            sha_resolvable = ok_sha
            if not ok_sha:
                result.warn(f"{path}:{lineno}: pre_fix_sha {pre_fix_sha} does not resolve in this "
                            "checkout (advisory — a shallow clone or an amend can cause this; "
                            "re-execution below is reported not_re_executed)")
        for file_key, entry in attacks.items():
            covered_files.add(file_key)
            if not isinstance(entry, dict):
                result.fail(f"{path}:{lineno}: attacks[{file_key!r}] is not an object")
                continue
            command = entry.get("command")
            recorded_hash = entry.get("hash")
            if not isinstance(command, str) or not command.strip():
                result.fail(f"{path}:{lineno}: attacks[{file_key!r}] has no `command`")
                continue
            if not (isinstance(recorded_hash, str) and re.fullmatch(r"[0-9a-f]{64}", recorded_hash)):
                result.fail(f"{path}:{lineno}: attacks[{file_key!r}] has no valid `hash`")
            deny = mod._r12_attack_command_denied(
                command, str(cwd),
                require_tracked_at=pre_fix_sha if sha_resolvable else None,
                project_dir=str(cwd), deadline=time.monotonic() + 5.0)
            if deny is not None:
                result.fail(f"{path}:{lineno}: attacks[{file_key!r}] command is denied: {deny}")

    missing = required_files - covered_files
    if missing:
        result.fail(f"{path}: omits {len(missing)} file(s) the union of open second-round BLOCK "
                    f"row(s) require, e.g. {sorted(missing)[:3]}")

    # Fix round 5 Z7: `unit_branch`/`residual_risk` presence, pair-reuse
    # denial and the execution-class requirement all run through the ONE
    # shared validator reader 1 (the hook) calls — never re-implemented
    # here, which is exactly how this file's own earlier gaps (accepting
    # an all-inspector record, a reused (command, hash) pair, a row with
    # no `residual_risk`/`unit_branch` at all) went uncaught.
    shape_why = mod._r12_anticipation_rejection([r for _, r in art_rows], [], check_attacks=True,
                                                 required_files=set())
    if shape_why is not None:
        result.fail(f"{path}: {shape_why}")

    # Re-execution against a REAL, independently checked-out `pre_fix_sha`
    # — ADVISORY ONLY, per M3': never a hard fail.
    for lineno, row in art_rows:
        pre_fix_sha = row.get("pre_fix_sha")
        attacks = row.get("attacks")
        if not (isinstance(pre_fix_sha, str) and pre_fix_sha and isinstance(attacks, dict)):
            continue
        ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{pre_fix_sha}^{{commit}}")
        if not ok_sha:
            result.warn(f"{path}:{lineno}: pre_fix_sha unreachable in this checkout — every attack "
                        "here is not_re_executed")
            continue
        with tempfile.TemporaryDirectory(prefix="r12-reader3-") as td:
            tmp_wt = Path(td) / "wt"
            ok_add, out_add = _git(cwd, "worktree", "add", "--detach", "-q", str(tmp_wt), pre_fix_sha)
            if not ok_add:
                result.warn(f"{path}:{lineno}: could not check out pre_fix_sha {pre_fix_sha} for "
                            f"advisory re-execution — {out_add} (not_re_executed)")
                continue
            try:
                for file_key, entry in sorted(attacks.items()):
                    if not isinstance(entry, dict):
                        continue
                    command = entry.get("command")
                    recorded_hash = entry.get("hash")
                    if not isinstance(command, str) or not command.strip():
                        continue
                    deny = mod._r12_attack_command_denied(
                        command, str(tmp_wt), require_tracked_at=pre_fix_sha,
                        project_dir=str(cwd), deadline=time.monotonic() + 5.0)
                    if deny is not None:
                        result.warn(f"{path}:{lineno}: attacks[{file_key!r}] not_re_executed "
                                    f"(denied in the detached checkout: {deny})")
                        continue
                    try:
                        proc = subprocess.run(["/bin/sh", "-c", command], cwd=str(tmp_wt),
                                               capture_output=True, text=True, timeout=120)
                    except subprocess.TimeoutExpired:
                        result.warn(f"{path}:{lineno}: attacks[{file_key!r}] not_re_executed "
                                    "(timed out)")
                        continue
                    stderr_lines = proc.stderr.splitlines()
                    stderr_line = stderr_lines[0] if stderr_lines else ""
                    actual_hash = mod._witness_hash(proc.returncode, proc.stdout, stderr_line)
                    if not (isinstance(recorded_hash, str) and actual_hash == recorded_hash):
                        result.warn(
                            f"{path}:{lineno}: attacks[{file_key!r}] mismatched at pre_fix_sha "
                            f"{pre_fix_sha} (advisory, never a failure — BSD/GNU userland and path "
                            "divergence between the lead's machine and CI's is a measured, known "
                            f"cause; recorded "
                            f"{recorded_hash[:12] if isinstance(recorded_hash, str) else recorded_hash}…, "
                            f"got {actual_hash[:12]}…)")
            finally:
                _git(cwd, "worktree", "remove", "--force", str(tmp_wt))


def _unit_allowlisted(unit_slug: str) -> bool:
    if not ALLOWLIST_PATH.exists():
        return False
    for line in ALLOWLIST_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == unit_slug:
            return True
    return False


def run_check(cwd: Path = REPO_ROOT) -> Result:
    result = Result()
    base_ref = os.environ.get("GITHUB_BASE_REF", "main")
    head_ref_env = os.environ.get("GITHUB_HEAD_REF")

    ok, changed, range_spec = compute_diff_context(cwd, base_ref)
    if not ok:
        result.warn("could not resolve the base ref / three-dot diff — treating as a no-op (no PR context)")
        return result

    reason = _no_op_reason(cwd, changed, range_spec)
    if reason is not None:
        print(f"check-rigor-record: no-op — {reason}")
        return result

    armed_paths = [p for p in changed if _matches_arming_glob(p)]
    if not armed_paths:
        print("check-rigor-record: not armed — diff touches none of crates/**, ci/**, .github/workflows/**")
        return result

    mod = _lib_module()
    if head_ref_env:
        head_branch = head_ref_env
    else:
        ok, head_branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
        if not ok:
            head_branch = "HEAD"
    unit_slug = mod.slugify(head_branch)

    if _unit_allowlisted(unit_slug):
        print(f"check-rigor-record: armed but {unit_slug!r} is on the shrink-only allowlist "
              f"({_display_path(ALLOWLIST_PATH)}) — no-op")
        return result

    print(f"check-rigor-record: ARMED — diff touches {len(armed_paths)} mechanism path(s), e.g. {armed_paths[:3]}")

    record_path = RIGOR_DIR / f"{unit_slug}.jsonl"
    ok, record_text = _git(cwd, "show", f"HEAD:docs/rigor/{unit_slug}.jsonl")
    if not ok:
        result.fail(f"no committed rigor record at docs/rigor/{unit_slug}.jsonl — export one with "
                    f"`python3 .claude/hooks/lead-gate-lib.py --export {unit_slug} "
                    f"> docs/rigor/{unit_slug}.jsonl` and commit it")
    else:
        rows = []
        for i, line in enumerate(record_text.splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                result.fail(f"docs/rigor/{unit_slug}.jsonl:{i + 1}: not valid JSON ({exc})")
        pressure_rows = [r for r in rows if isinstance(r, dict) and r.get("agent_type") == "pressure-tester"]
        if not pressure_rows:
            result.fail(f"docs/rigor/{unit_slug}.jsonl carries no pressure-tester row — "
                        "design-before-mechanism (.claude/agents/lead.md:41) requires one before "
                        "a non-local fix; dispatch it and re-export the record")
        # Check 3 (ancestry) — ADVISORY ONLY, never fails the check.
        for r in rows:
            if not isinstance(r, dict):
                continue
            sha = r.get("head_sha")
            if not isinstance(sha, str) or not sha:
                continue
            ok_sha, _ = _git(cwd, "rev-parse", "--verify", f"{sha}^{{commit}}")
            if not ok_sha:
                result.warn(f"docs/rigor/{unit_slug}.jsonl: head_sha {sha} does not resolve in this "
                            "checkout (advisory — an amend or a shallow clone can cause this)")
                continue
            ok_anc, _ = _git(cwd, "merge-base", "--is-ancestor", sha, "HEAD")
            if not ok_anc:
                result.warn(f"docs/rigor/{unit_slug}.jsonl: head_sha {sha} is not an ancestor of HEAD "
                            "(advisory — the amend-after-verification workflow does this on the honest path)")

        # esc-lead-gate-R12 READER 3 (M3') — armed whenever the exported
        # record carries an open second-round BLOCK row not on the
        # shrink-only grandfather list; a shape-only hard fail, re-execution
        # ADVISORY.
        check_anticipation_witnesses(cwd, unit_slug, [r for r in rows if isinstance(r, dict)], result)

        # esc-lead-gate-R12 fix round 3 item 8a READER 3 — armed only when
        # ci/lead-gate-required-commands.txt is itself committed at HEAD;
        # shape+value only, never re-executed.
        check_required_gates(cwd, unit_slug, result)

    contract_paths = [p for p in changed if _matches_contract_glob(p)]
    if not contract_paths:
        result.fail("diff arms (touches crates/**/ci/**/.github/workflows/**) but adds/modifies no "
                    "committed mechanism contract under docs/plans/*/proposals/** or docs/rigor/contracts/**")
    else:
        ok, all_contracts_out = _git(cwd, "ls-files", "docs/plans/*/proposals", "docs/rigor/contracts")
        all_contracts = [p for p in all_contracts_out.splitlines() if p.strip()] if ok else contract_paths
        for cp in contract_paths:
            ok, text = _git(cwd, "show", f"HEAD:{cp}")
            if not ok:
                result.fail(f"{cp}: named in the diff but does not read at HEAD (deleted?)")
                continue
            check_path_line_citations(cwd, cp, text, result)
            check_introducing_commit_ancestor(cwd, cp, result)
            check_not_near_identical(cwd, cp, text, all_contracts, result)

    return result


def check_required_gates(cwd: Path, unit_slug: str, result: Result) -> None:
    """esc-lead-gate-R12 fix round 3 item 8a, READER 3: when the repo
    commits `ci/lead-gate-required-commands.txt` (human-amend-only; this
    unit's own diff need not touch it), the GOVERNING row of the exported
    anticipation record must carry a `gates` object naming EVERY committed
    line VERBATIM, each with an integer `rc`, and every `rc` must be `0` —
    the COMMITTED record is expected to reflect the fix's own verified
    state, never a transient broken-tip snapshot recorded mid-round. Shape
    and value only — never re-executed (these are already CI jobs
    elsewhere in `.github/workflows/`).

    Fix round 4 Z2 / fix round 5 Z5: the governing row is selected ORDER-
    INDEPENDENTLY, never by `rows[-1]` (append position). `cmd_export_
    anticipation` dumps every `<slug>.anticipation.*.json` artifact still
    on disk, `sorted(sdir.iterdir())` — i.e. sorted by FILENAME, a tip
    sha, which is pseudorandom hex and carries no chronological meaning;
    an older round's artifact can sort AFTER a newer round's and land on
    the last line of the committed export. Since fix round 5, the SAME
    exporter stamps `ts` (the artifact FILE's own mtime) and `head_sha`
    (the artifact's own `pre_fix_sha`) on every row it emits — so
    `_row_head`/`_row_ts` below select the row whose own `head_sha`
    matches this checkout's actual `HEAD` when one does (the case where
    the export was captured at the exact commit reader 3 is validating);
    when none does — the common case, since a pre-fix witness by
    construction predates the commit it is validated against — every row
    is eligible. Within whichever pool applies, the GREATEST `ts`
    governs, never the row nearest the end of the file — and when the
    pool holds >=2 candidate rows and ANY of them lacks `ts` (a
    hand-typed or pre-Z5 record, never one the real exporter produced),
    this FAILS LOUDLY naming the ambiguity rather than silently falling
    back to `rows[0]`/append order.

    Fix round 5 Z7: the gates SHAPE/VALUE check itself is the SAME shared
    `_r12_gates_shape_rejection` reader 1 and reader 2 call — never a
    second, independently hand-rolled implementation that can drift from
    theirs (this is exactly the bug three earlier readers of this file
    found: a hand-rolled loop here accepted `{"rc": "1"}`/`{"rc": true}`/
    non-dict `gates` entries that the shared function has always denied)."""
    ok_req, req_text = _git(cwd, "show", "HEAD:ci/lead-gate-required-commands.txt")
    if not ok_req:
        # Fix round 5 Z4: MISSING is a hard FAIL here, never a silent
        # return — a missing committed gate-command list must never
        # disarm item 8a with zero CI signal.
        result.fail("ci/lead-gate-required-commands.txt does not read at HEAD in this checkout — "
                    "item 8a's committed, human-amend-only gate-command list must be present "
                    "(esc-lead-gate-R12 fix round 5 Z4)")
        return
    required_commands: list[str] = []
    for line in req_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split("  #", 1)[0].rstrip()
        if command:
            required_commands.append(command)
    if not required_commands:
        # Fix round 5 Z4: EMPTY/all-comment is likewise a hard FAIL.
        result.fail("ci/lead-gate-required-commands.txt exists but names no command line (empty "
                    "or all-comment) — item 8a's gate-command list must never silently disarm "
                    "(esc-lead-gate-R12 fix round 5 Z4)")
        return
    path = f"docs/rigor/{unit_slug}.anticipation.jsonl"
    ok, text = _git(cwd, "show", f"HEAD:{path}")
    if not ok or not text.strip():
        # No anticipation record for THIS unit at all -- structurally
        # nothing to check item 8a against; `check_anticipation_witnesses`
        # already fails the missing-record shape when IT is armed (an open
        # second-round BLOCK), which is the only case that requires one.
        return
    rows: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    if not rows:
        return

    ok_head, head_now_raw = _git(cwd, "rev-parse", "HEAD")
    head_now = head_now_raw.strip() if ok_head else None

    def _row_head(r: dict) -> str | None:
        h = r.get("head_sha")
        return h if isinstance(h, str) and h else None

    def _row_ts(r: dict) -> str:
        ts = r.get("ts")
        return ts if isinstance(ts, str) else ""

    matching = [r for r in rows if head_now and _row_head(r) == head_now]
    pool = matching if matching else rows

    if len(pool) >= 2 and any(not (isinstance(r.get("ts"), str) and r.get("ts")) for r in pool):
        result.fail(
            f"{path}: {len(pool)} candidate anticipation row(s) carry no reliable ordering "
            "evidence (at least one has no `ts`) -- the GOVERNING row is AMBIGUOUS; re-export "
            "with `lead-gate-lib.py --export-anticipation` (which stamps `ts`/`head_sha` on "
            "every row since fix round 5) and commit the result (esc-lead-gate-R12 fix round 5 Z5)"
        )
        return
    governing = max(pool, key=_row_ts)

    mod = _lib_module()
    gates_why = mod._r12_gates_shape_rejection(f"{path}: the governing row", governing.get("gates"),
                                                required_commands, judge_rc=True)
    if gates_why is not None:
        result.fail(gates_why)


def check_allowlist_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-allowlist-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-allowlist-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = set()
    if ALLOWLIST_PATH.exists():
        for line in ALLOWLIST_PATH.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                current.add(s)
    rel = ALLOWLIST_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-allowlist-only-shrinks: OK (bootstrap) — origin/main has no {rel} yet; "
              f"this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("rigor-record-allowlist-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\nrigor-record-allowlist-only-shrinks: this branch adds a NEW exemption. The "
              "allowlist may only shrink — a genuinely new exemption is a human-reviewed decision, "
              "made on main directly, never an autonomous addition on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-allowlist-only-shrinks: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


def check_r12_grandfather_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    """esc-lead-gate-R12 (M3'): the SAME shrink-only ratchet as
    `check_allowlist_only_shrinks`, applied to `R12_GRANDFATHER_PATH`
    instead — a separate check because the two exemptions arm on
    different diffs and are read by different functions."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = set()
    if R12_GRANDFATHER_PATH.exists():
        for line in R12_GRANDFATHER_PATH.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                current.add(s)
    rel = R12_GRANDFATHER_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-r12-grandfather-only-shrinks: OK (bootstrap) — origin/main has no {rel} "
              f"yet; this branch's {len(current)} entries establish the baseline.")
        return 0
    base = {s.strip() for s in base_text.splitlines() if s.strip() and not s.strip().startswith("#")}
    added = current - base
    if added:
        print("rigor-record-r12-grandfather-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(added):
            print(f"  + {e}", file=sys.stderr)
        print("\nrigor-record-r12-grandfather-only-shrinks: this branch adds a NEW exemption. The "
              "grandfather list may only shrink — a genuinely new exemption (an in-flight unit "
              "whose BLOCK predates fix round 1) is a human-reviewed decision, made on main "
              "directly, never an autonomous addition on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-r12-grandfather-only-shrinks: OK — {len(current)} entries "
          f"({len(base) - len(current)} shrunk vs origin/main).")
    return 0


def _parse_required_commands(text: str) -> set[str]:
    """The SAME parse `_r12_required_commands_or_deny` in lead-gate-lib.py
    applies — `#`-comment/blank lines skipped, a trailing `  # ...`
    annotation stripped — reduced to a SET of command strings (never the
    annotation, which a committer may re-measure/reword without that
    being a real removal)."""
    out: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split("  #", 1)[0].rstrip()
        if command:
            out.add(command)
    return out


def check_required_commands_only_shrinks(cwd: Path = REPO_ROOT) -> int:
    """esc-lead-gate-R12 fix round 5 Z4(b): the OPPOSITE polarity from
    `check_allowlist_only_shrinks`/`check_r12_grandfather_only_shrinks` —
    `REQUIRED_COMMANDS_PATH` names REQUIRED gates, so this check's job is
    to CATCH a SHRINK (a committed command REMOVED, weakening item 8a)
    and FAIL it; adding a new required command, or leaving the set
    unchanged, always passes. BOOTSTRAP arm: `origin/main` carrying no
    such file yet (this PR is the one introducing it) establishes the
    baseline instead of failing."""
    ok, _ = _git(cwd, "fetch", "--quiet", "origin", "main")
    if not ok:
        print("rigor-record-required-commands-only-shrinks: FAIL — git fetch origin main failed", file=sys.stderr)
        return 1
    ok, _ = _git(cwd, "rev-parse", "--verify", "origin/main")
    if not ok:
        print("rigor-record-required-commands-only-shrinks: FAIL — origin/main does not resolve", file=sys.stderr)
        return 1
    current = _parse_required_commands(REQUIRED_COMMANDS_PATH.read_text()) if REQUIRED_COMMANDS_PATH.exists() else set()
    rel = REQUIRED_COMMANDS_PATH.relative_to(cwd).as_posix()
    ok, base_text = _git(cwd, "show", f"origin/main:{rel}")
    if not ok:
        print(f"rigor-record-required-commands-only-shrinks: OK (bootstrap) — origin/main has no "
              f"{rel} yet; this branch's {len(current)} command(s) establish the baseline.")
        return 0
    base = _parse_required_commands(base_text)
    removed = base - current
    if removed:
        print("rigor-record-required-commands-only-shrinks: FAIL", file=sys.stderr)
        for e in sorted(removed):
            print(f"  - {e}", file=sys.stderr)
        print("\nrigor-record-required-commands-only-shrinks: this branch REMOVES a committed "
              "required gate command. The list may only grow (or stay the same) — removing a "
              "required gate weakens item 8a and is a human-reviewed decision, made on main "
              "directly, never an autonomous removal on a swarm branch.", file=sys.stderr)
        return 1
    print(f"rigor-record-required-commands-only-shrinks: OK — {len(current)} command(s) "
          f"({len(current) - len(base)} added vs origin/main).")
    return 0


# ==========================================================================
# --self-test — hermetic fixtures, the check_lead_gate.py harness pattern:
# a real `origin` remote + a real feature-branch clone, run against the
# REAL run_check()/check_allowlist_only_shrinks(), never a reimplementation.
# ==========================================================================

class Failure(Exception):
    pass


def _assert(cond: bool, label: str, detail: str = "") -> None:
    if not cond:
        raise Failure(f"{label}: {detail}")


def _sh(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    _assert(proc.returncode == 0, "git fixture setup", f"git {' '.join(args)} failed: {proc.stderr}")
    return proc.stdout.strip()


_RR_BASELINE_REQUIRED_COMMAND = "python3 ci/scripts/check_swarm_bijection.py"

_FIXTURE_ENV = {
    "GIT_AUTHOR_NAME": "rigor-fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "rigor-fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def _pr_repo(tmp: Path) -> tuple[Path, Path]:
    """A real `origin` repo (its own `main`, one seed commit) and a real
    clone (`work`) with `origin` wired up — `git fetch origin main` in
    `work` resolves `origin/main` exactly as a real CI checkout would.
    Copies THIS run's own `.claude/hooks/lead-gate-lib.py` and
    `ci/scripts/check_rigor_record.py` into `work` (never `.claude/hooks/`
    from the real repo directly — a throwaway copy, per this file's own
    module doc) so `_lib_module()`/self-invocation resolve inside the
    fixture, not the real tree."""
    origin = tmp / "origin"
    work = tmp / "work"
    origin.mkdir()
    env = dict(os.environ)
    env.update(_FIXTURE_ENV)
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True, env=env)
    subprocess.run(["git", "-C", str(origin), "config", "commit.gpgsign", "false"], check=True)
    # Fixture-only: `origin` is a NON-bare repo so its own working tree can
    # be read directly for setup; allow `work` to push updates to its
    # currently-checked-out `main` (git denies this by default) — never a
    # real-repo concern, since the real `origin` in CI is GitHub itself.
    subprocess.run(["git", "-C", str(origin), "config", "receive.denyCurrentBranch", "updateInstead"], check=True)
    (origin / "README.md").write_text("seed\n")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "seed"], check=True, env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "config", "commit.gpgsign", "false"], check=True)
    # Mirror the real tree's own module layout inside the fixture: this
    # script and lead-gate-lib.py at their real relative paths, so
    # REPO_ROOT/`_lib_module()` resolve inside `work`, never the real repo.
    (work / ".claude" / "hooks").mkdir(parents=True)
    (work / "ci" / "scripts").mkdir(parents=True)
    (work / "docs" / "rigor" / "contracts").mkdir(parents=True)
    (work / "docs" / "plans" / "99-fixture" / "proposals").mkdir(parents=True)
    lib_text = LEAD_GATE_LIB.read_text() if LEAD_GATE_LIB.exists() else _MINIMAL_SLUGIFY_STUB
    (work / ".claude" / "hooks" / "lead-gate-lib.py").write_text(lib_text)
    (work / "ci" / "scripts" / "check_rigor_record.py").write_text(Path(__file__).read_text())
    # Fix round 5 Z4: a missing/empty/all-comment required-commands file is
    # now a hard FAIL in `check_required_gates` whenever an anticipation
    # record exists at all -- every fixture gets a baseline, non-empty file
    # by default so unrelated fixtures never incidentally exercise item
    # 8a's own FAIL arm; `_rr_gates_commit` (RR14-17) overwrites this with
    # its own custom line, and RR12c/d/e/g/h's own anticipation rows carry
    # a matching `gates` entry.
    (work / "ci" / "lead-gate-required-commands.txt").write_text(
        f"{_RR_BASELINE_REQUIRED_COMMAND}  # measured ~0.0s\n")
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", "scaffold"], check=True, env=env)
    subprocess.run(["git", "-C", str(work), "push", "-q", "origin", "HEAD:main"], check=True)
    subprocess.run(["git", "-C", str(work), "checkout", "-q", "-b", "feat/rr-fixture"], check=True)
    return origin, work


# A minimal, self-contained fallback if this fixture ever runs with no real
# lead-gate-lib.py reachable (never used against the real tree — only a
# defensive stub so the fixture harness itself cannot silently pass by
# accident when the real file is missing).
_MINIMAL_SLUGIFY_STUB = (
    "import re\n"
    "def slugify(branch):\n"
    "    return re.sub(r'[^A-Za-z0-9._-]', '_', branch.strip()) or 'UNBOUND'\n"
)


def _commit(work: Path, message: str, files: dict[str, str], swarm: bool = True) -> str:
    for rel, content in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    full_msg = message + ("\n\nCo-Authored-By: Claude Fixture <noreply@anthropic.com>" if swarm else "")
    env = dict(os.environ)
    env.update(_FIXTURE_ENV)
    subprocess.run(["git", "-C", str(work), "commit", "-q", "-m", full_msg], check=True, env=env)
    return _sh(work, "rev-parse", "HEAD")


def _run_check_in(work: Path, env_overrides: dict | None = None) -> Result:
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_BASE_REF": "main", "GITHUB_HEAD_REF": "feat/rr-fixture"}
    if env_overrides:
        env.update(env_overrides)
    old = {}
    for k, v in env.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        global LEAD_GATE_LIB
        real_lib = LEAD_GATE_LIB
        LEAD_GATE_LIB = work / ".claude" / "hooks" / "lead-gate-lib.py"
        try:
            return run_check(work)
        finally:
            LEAD_GATE_LIB = real_lib
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_VALID_CONTRACT = "# A mechanism contract\n\nCites `docs/README-fixture.md:1` which exists.\n"


def fixture_rr1_not_armed_docs_only() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "docs: a docs-only change", {"docs/only.md": "hello\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR1", f"a docs-only diff must not arm: {r.failures}")


def fixture_rr2_armed_no_record() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        r = _run_check_in(work)
        _assert(not r.ok(), "RR2", "armed with no rigor record must FAIL")
        _assert(any("no committed rigor record" in f for f in r.failures), "RR2",
                f"reason must name the missing record: {r.failures}")


def fixture_rr3_record_no_pressure_row() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "adversarial-audit", "verdict": "PASS"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR3", "a record with no pressure-tester row must FAIL")
        _assert(any("no pressure-tester row" in f for f in r.failures), "RR3", f"{r.failures}")


def fixture_rr4_no_contract_file() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR4", "armed with a record+pressure row but no contract file must FAIL")
        _assert(any("no committed mechanism contract" in f for f in r.failures), "RR4", f"{r.failures}")


def fixture_rr5_full_disclosure_allows() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR5", f"full disclosure must ALLOW: {r.failures}")


def fixture_rr6_bad_citation_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        bad_contract = "# Contract\n\nCites `docs/README-fixture.md:99` which does not have 99 lines.\n"
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": bad_contract,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR6", "a citation past the file's own line count must FAIL")
        _assert(any("has only" in f for f in r.failures), "RR6", f"{r.failures}")


def fixture_rr7_near_identical_contract_fails() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        # Commit an EXISTING contract on main first (via a swarm commit on
        # the feature branch itself, at the SAME path a second file will
        # copy — near-identical after whitespace normalization).
        _commit(work, "docs: an existing contract", {
            "docs/rigor/contracts/other-unit.md": "# A mechanism contract\n\nCites `docs/README-fixture.md:1` which exists.\n",
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": "  # A mechanism contract  \n\n\nCites `docs/README-fixture.md:1` which exists.\n",
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR7", "a whitespace-only variant of an existing contract must FAIL as near-identical")
        _assert(any("near-identical" in f for f in r.failures), "RR7", f"{r.failures}")


def fixture_rr8_noop_shapes() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script, human-authored", {"ci/scripts/probe.py": "print('x')\n"}, swarm=False)
        r = _run_check_in(work)
        _assert(r.ok(), "RR8a", f"a human-authored-only range must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "chore: bump crate version", {"crates/foo/Cargo.toml": "[package]\nversion=\"0.2.0\"\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR8b", f"a release-shaped (Cargo.toml-only) diff must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        r = _run_check_in(work, {"GITHUB_ACTOR": "dependabot[bot]"})
        _assert(r.ok(), "RR8c", f"a dependabot actor must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        _commit(work, 'Revert "ci: touch a gate script"', {"ci/scripts/probe.py": "orig\n"})
        r = _run_check_in(work)
        _assert(r.ok(), "RR8d", f"a revert HEAD commit must no-op: {r.failures}")

    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        r = _run_check_in(work, {"GITHUB_EVENT_NAME": "push"})
        _assert(r.ok(), "RR8e", f"a non-pull_request event must no-op: {r.failures}")


def fixture_rr9_ancestry_advisory_only() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED",
                           "head_sha": "cafef00d1234567890abcdef1234567890abcdef"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR9", f"an unresolvable head_sha must WARN, never FAIL: {r.failures}")
        _assert(any("does not resolve" in w for w in r.warnings), "RR9",
                f"the unresolvable sha must be reported as a warning: {r.warnings}")


def fixture_rr10_allowlisted_unit_noop() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        (work / "ci" / "scripts" / "rigor_record_allowlist.txt").write_text("# fixture allowlist\nfeat_rr-fixture\n")
        _commit(work, "ci: touch a gate script", {"ci/scripts/probe.py": "print('x')\n"})
        global ALLOWLIST_PATH
        real_allowlist = ALLOWLIST_PATH
        ALLOWLIST_PATH = work / "ci" / "scripts" / "rigor_record_allowlist.txt"
        try:
            r = _run_check_in(work)
        finally:
            ALLOWLIST_PATH = real_allowlist
        _assert(r.ok(), "RR10", f"an allowlisted unit must no-op even though armed: {r.failures}")


def fixture_rr11_allowlist_only_shrinks() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        allow_rel = "ci/scripts/rigor_record_allowlist.txt"
        _commit(work, "seed allowlist", {allow_rel: "unit-a\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global ALLOWLIST_PATH
        real_allowlist = ALLOWLIST_PATH
        ALLOWLIST_PATH = work / allow_rel
        try:
            rc = check_allowlist_only_shrinks(work)
            _assert(rc == 0, "RR11a", "an unchanged allowlist must pass the shrink-only ratchet")
            (work / allow_rel).write_text("unit-a\nunit-b\n")
            rc = check_allowlist_only_shrinks(work)
            _assert(rc != 0, "RR11b", "adding a NEW entry (not yet on origin/main) must FAIL the ratchet")
        finally:
            ALLOWLIST_PATH = real_allowlist


def fixture_rr12a_no_open_block_row_is_a_noop() -> None:
    """esc-lead-gate-R12 READER 3 (M3'): a record with NO open second-round
    BLOCK row at all (only a pressure-tester row) never arms this check —
    the SAME full-disclosure record RR5 already proves ALLOWS stays
    ALLOWED, with no `docs/rigor/<slug>.anticipation.jsonl` carried
    either."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12a", f"no open second-round BLOCK row must be a no-op (still ALLOW): {r.failures}")


def fixture_rr12b_open_block_no_anticipation_file_fails() -> None:
    """An open `adversarial-audit` BLOCK row is present, but NO
    `docs/rigor/<slug>.anticipation.jsonl` at all — HARD FAILS, naming the
    `--export-anticipation` remedy (never a retroactive pass just because
    the artifact happens to be absent)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR12b", "an open second-round BLOCK row with no anticipation file must FAIL")
        _assert(any("--export-anticipation" in f for f in r.failures), "RR12b", f"{r.failures}")


def fixture_rr12c_shape_allows_despite_unresolvable_pre_fix_sha() -> None:
    """A correctly-shaped anticipation record (covers the required file,
    denylist-clean command) ALLOWS even though `pre_fix_sha` does not
    resolve in this checkout (a shallow-clone-shaped repro) — the HARD
    requirement is shape alone; re-execution is advisory and only WARNS
    when it cannot even attempt reproduction."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok\n", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12c", f"a correctly-shaped record must ALLOW despite an unresolvable "
                                  f"pre_fix_sha: {r.failures}")
        _assert(any("does not resolve" in w for w in r.warnings), "RR12c",
                f"the unresolvable pre_fix_sha must still be visible as a warning: {r.warnings}")


def fixture_rr12d_real_reproducing_witness_allows() -> None:
    """The SAME shape as RR12c, but `pre_fix_sha` is a REAL commit this
    check independently checks out and the command REPRODUCES there —
    ALLOWS, with no mismatch warning for this attack."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok\n", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12d", f"a real reproducing witness at pre_fix_sha must ALLOW: {r.failures}")
        _assert(not any("mismatched at pre_fix_sha" in w for w in r.warnings), "RR12d",
                f"a genuinely reproducing witness must not be reported mismatched: {r.warnings}")


def fixture_rr12e_real_nonreproducing_witness_warns_never_fails() -> None:
    """The SAME REAL `pre_fix_sha` checkout as RR12d, but the recorded
    `hash` does NOT match what re-executing the command actually produces
    there — WARNS naming the mismatch, but NEVER FAILS (M3': re-execution
    is advisory only, since a lead's machine and CI's userland can
    genuinely diverge on some commands without the mechanism itself having
    changed)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha,
            "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "0" * 64}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12e", f"a non-reproducing witness must WARN, never FAIL: {r.failures}")
        _assert(any("mismatched at pre_fix_sha" in w for w in r.warnings), "RR12e",
                f"the mismatch must still be visible as a warning: {r.warnings}")


def fixture_rr12f_grandfathered_unit_no_file_allows() -> None:
    """An open second-round BLOCK row with NO anticipation file at all is a
    HARD FAIL (RR12b) — UNLESS the unit is on the shrink-only R12
    grandfather list, in which case it ALLOWS (advisory warning only)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        (work / "ci" / "scripts" / "rigor_record_r12_grandfather.txt").write_text("# fixture\nfeat_rr-fixture\n")
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        global R12_GRANDFATHER_PATH
        real_path = R12_GRANDFATHER_PATH
        R12_GRANDFATHER_PATH = work / "ci" / "scripts" / "rigor_record_r12_grandfather.txt"
        try:
            r = _run_check_in(work)
        finally:
            R12_GRANDFATHER_PATH = real_path
        _assert(r.ok(), "RR12f", f"a grandfathered unit must ALLOW despite no anticipation file: {r.failures}")


def fixture_rr12g_tracked_bash_path_allows() -> None:
    """M3': `bash <path>` is admitted only for paths git-TRACKED AT
    `pre_fix_sha` — a script added IN that same commit is tracked there,
    and the shape check (plus a real, matching re-execution) ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        _commit(work, "add probe.sh, tracked here", {"probe.sh": "#!/bin/sh\nprintf ok\n"})
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")
        mod = _lib_module()
        witness = mod._witness_hash(0, "ok", "")
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha,
            "attacks": {"a.py": {"command": "bash probe.sh", "hash": witness}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(r.ok(), "RR12g", f"a bash path tracked at pre_fix_sha must ALLOW: {r.failures}")


def fixture_rr12h_untracked_bash_path_fails_shape() -> None:
    """M3': the SAME `bash probe.sh` command, but `pre_fix_sha` names a
    commit BEFORE `probe.sh` was ever added — untracked there — HARD
    FAILS the shape check, naming it; tracked-ness buys reviewability, not
    safety, but CI must never execute an untracked lead-written file from
    a detached checkout."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        pre_fix_sha = _sh(work, "rev-parse", "HEAD")  # BEFORE probe.sh exists
        _commit(work, "add probe.sh AFTER pre_fix_sha", {"probe.sh": "#!/bin/sh\nprintf ok\n"})
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        anticipation_row = json.dumps({
            "unit_branch": "feat/rr-fixture", "pre_fix_sha": pre_fix_sha,
            "attacks": {"a.py": {"command": "bash probe.sh", "hash": "a" * 64}},
            "residual_risk": "fixture residual",
            "gates": {_RR_BASELINE_REQUIRED_COMMAND: {"rc": 0}},
        })
        _commit(work, "ci: touch a gate script", {
            "ci/scripts/probe.py": "print('x')\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": anticipation_row + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        })
        r = _run_check_in(work)
        _assert(not r.ok(), "RR12h", "a bash path NOT tracked at pre_fix_sha must FAIL the shape check")
        _assert(any("is not git-TRACKED" in f for f in r.failures), "RR12h", f"{r.failures}")


def fixture_rr13_r12_grandfather_only_shrinks() -> None:
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        gf_rel = "ci/scripts/rigor_record_r12_grandfather.txt"
        _commit(work, "seed r12 grandfather list", {gf_rel: "unit-a\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global R12_GRANDFATHER_PATH
        real_path = R12_GRANDFATHER_PATH
        R12_GRANDFATHER_PATH = work / gf_rel
        try:
            rc = check_r12_grandfather_only_shrinks(work)
            _assert(rc == 0, "RR13a", "an unchanged grandfather list must pass the shrink-only ratchet")
            (work / gf_rel).write_text("unit-a\nunit-b\n")
            rc = check_r12_grandfather_only_shrinks(work)
            _assert(rc != 0, "RR13b", "adding a NEW entry (not yet on origin/main) must FAIL the ratchet")
        finally:
            R12_GRANDFATHER_PATH = real_path


def fixture_rr19_required_commands_only_shrinks() -> None:
    """fix round 5 Z4(b): the OPPOSITE polarity from RR13's own ratchet —
    `ci/lead-gate-required-commands.txt` names REQUIRED gates, so ADDING a
    line passes (tightening) and REMOVING a committed line FAILS
    (weakening item 8a with zero human review)."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        origin, work = _pr_repo(Path(td))
        rc_rel = "ci/lead-gate-required-commands.txt"
        _commit(work, "seed required-commands file", {rc_rel: "python3 ci/scripts/probe_a.py  # measured ~0.1s\n"})
        _sh(work, "push", "-q", "origin", "HEAD:main")
        _sh(work, "fetch", "-q", "origin", "main")
        global REQUIRED_COMMANDS_PATH
        real_path = REQUIRED_COMMANDS_PATH
        REQUIRED_COMMANDS_PATH = work / rc_rel
        try:
            rc = check_required_commands_only_shrinks(work)
            _assert(rc == 0, "RR19a", "an unchanged required-commands file must pass the ratchet")
            (work / rc_rel).write_text(
                "python3 ci/scripts/probe_a.py  # measured ~0.1s\n"
                "python3 ci/scripts/probe_b.py  # measured ~0.1s\n")
            rc = check_required_commands_only_shrinks(work)
            _assert(rc == 0, "RR19b", "ADDING a new required command must PASS the ratchet (tightening)")
            (work / rc_rel).write_text("# nothing but comments\n")
            rc = check_required_commands_only_shrinks(work)
            _assert(rc != 0, "RR19c", "REMOVING a committed required command must FAIL the ratchet")
        finally:
            REQUIRED_COMMANDS_PATH = real_path


def _rr_gates_commit(work: Path, extra_files: dict[str, str], gates: dict | None) -> None:
    """Shared setup for RR14-RR16: a committed `ci/lead-gate-required-
    commands.txt` (one line) plus an anticipation record whose single row
    carries `gates` (or omits it, when `gates is None`)."""
    pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
    block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                             "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                             "class_enumeration": ["a.py:1"]})
    art: dict = {
        "unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
        "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
        "residual_risk": "fixture residual",
    }
    if gates is not None:
        art["gates"] = gates
    files = {
        "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
        "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
        "docs/rigor/feat_rr-fixture.anticipation.jsonl": json.dumps(art) + "\n",
        "docs/README-fixture.md": "line one\n",
        "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
    }
    files.update(extra_files)
    _commit(work, "ci: touch a gate script", files)


def fixture_rr14_missing_gates_fails() -> None:
    """item 8a READER 3: `ci/lead-gate-required-commands.txt` is
    committed, but the exported anticipation record's row carries no
    `gates` object at all -- HARD FAIL."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates=None)
        r = _run_check_in(work)
        _assert(not r.ok(), "RR14", "a missing `gates` object must FAIL when required-commands.txt is committed")
        _assert(any("carries no `gates` object" in f for f in r.failures), "RR14", f"{r.failures}")


def fixture_rr15_complete_gates_rc_zero_allows() -> None:
    """POSITIVE CONTROL — green at base by construction: guards the reader
    against over-refusal (a shape-complete, fully green governing `gates`
    row must ALLOW). The RED half is RR14/RR16. item 8a READER 3: the
    satisfiable case -- every committed line present with `rc == 0`
    ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates={"python3 ci/scripts/probe.py": {"rc": 0}})
        r = _run_check_in(work)
        _assert(r.ok(), "RR15", f"a fully green gates object must ALLOW: {r.failures}")


def fixture_rr16_nonzero_rc_fails() -> None:
    """item 8a READER 3: a `gates` entry recording a non-zero `rc` in the
    COMMITTED record HARD FAILS -- the exported record is expected to
    reflect the fix's own verified state."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        _rr_gates_commit(work, {}, gates={"python3 ci/scripts/probe.py": {"rc": 1}})
        r = _run_check_in(work)
        _assert(not r.ok(), "RR16", "a non-zero committed rc must FAIL")
        _assert(any("recorded rc=1" in f for f in r.failures), "RR16", f"{r.failures}")


def _real_anticipation_export(work: Path, slug: str, artifacts: list[dict],
                               mtimes: list[float]) -> str:
    """Fix round 5 Z5: runs the REAL `lead-gate-lib.py --export-anticipation`
    entry point (a subprocess, `CLAUDE_PROJECT_DIR=work` — the exact
    command the lead runs) against real `.jammi/gate-state/<slug>.
    anticipation.<tip>.json` artifact files, one per `artifacts[i]`, whose
    mtime is set to `mtimes[i]` — proving the fixture exercises the
    EXPORTED shape (real `ts`/`head_sha` stamped from the artifact file's
    own mtime/`pre_fix_sha`), never a hand-typed simulation of it."""
    sdir = work / ".jammi" / "gate-state"
    sdir.mkdir(parents=True, exist_ok=True)
    for art, mtime in zip(artifacts, mtimes):
        tip = art["pre_fix_sha"]
        p = sdir / f"{slug}.anticipation.{tip}.json"
        p.write_text(json.dumps(art))
        os.utime(p, (mtime, mtime))
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = str(work)
    proc = subprocess.run(
        [sys.executable, str(work / ".claude" / "hooks" / "lead-gate-lib.py"),
         "--export-anticipation", slug],
        capture_output=True, text=True, env=env, timeout=10,
    )
    _assert(proc.returncode == 0, "real export setup", f"--export-anticipation failed: {proc.stderr}")
    return proc.stdout


def fixture_rr17_interleaved_export_selects_by_ts_not_position() -> None:
    """fix round 4 Z2 / fix round 5 Z5: `cmd_export_anticipation` sorts
    artifacts by FILENAME (a tip sha -- pseudorandom hex, no chronological
    meaning), so an OLDER round's row can land on the LAST line of the
    committed export while a NEWER, green row sits earlier in the file.
    Exercises the REAL exporter (`_real_anticipation_export`) against two
    real artifact files whose mtimes are set OLDER-first/NEWER-second in
    wall-clock time but whose FILENAMES sort in the OPPOSITE order (the
    older, broken `rc=1` artifact's tip sorts AFTER the newer, green
    `rc=0` artifact's tip) -- the governing row must be the one with the
    greatest `ts` (the real, exported mtime), never the one nearest the
    end of the file; the fix ALLOWS."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        sha0 = _sh(work, "rev-parse", "HEAD")
        sha1 = _commit(work, "seed an intermediate ancestor tip", {"docs/README-fixture2.md": "seed2\n"})
        # tip_new sorts BEFORE tip_old alphabetically ("0..." < "f...") --
        # the OPPOSITE of chronological order, exactly fix round 4 F1's
        # own bug shape (an older round's artifact filename can sort
        # AFTER a newer one).
        tip_new, tip_old = "0" * 40, "f" * 40
        art_new = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": tip_new,
                   "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                   "residual_risk": "fixture residual",
                   "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
        art_old = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": tip_old,
                   "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                   "residual_risk": "fixture residual",
                   "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
        now = time.time()
        exported = _real_anticipation_export(work, "feat_rr-fixture", [art_old, art_new],
                                              [now - 3600.0, now])
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        files = {
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": exported,
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        }
        _commit(work, "ci: touch a gate script (interleaved REAL export, older head last)", files)
        r = _run_check_in(work)
        _assert(r.ok(), "RR17", f"the greatest-ts (real, exported) row (rc=0) must govern despite "
                                  f"the older-ts row (rc=1) sorting LAST by filename: {r.failures}")


def fixture_rr18_ambiguous_pool_without_ts_fails_loudly() -> None:
    """fix round 5 Z5: the PRE-fix-round-5 shape -- two candidate rows,
    NEITHER carrying `ts`/`head_sha` at all (what `cmd_export_anticipation`
    produced before it started stamping them) -- is now a LOUD FAIL naming
    the ambiguity, never a silent `rows[0]`/append-order pick. RED at
    daebd948 by the executed probe: the old code's `max(pool, key=_row_ts)`
    with an all-empty `_row_ts` silently returned the FIRST row for every
    such pool, regardless of which one was actually current."""
    with tempfile.TemporaryDirectory(prefix="rr-fixture-") as td:
        _origin, work = _pr_repo(Path(td))
        pressure_row = json.dumps({"ts": "2026-01-01T00:00:00Z", "agent_type": "pressure-tester", "verdict": "PROCEED"})
        block_row = json.dumps({"ts": "2026-01-01T00:01:00Z", "agent_type": "adversarial-audit",
                                 "verdict": "BLOCK", "finding_locations": ["a.py:1"],
                                 "class_enumeration": ["a.py:1"]})
        row_a = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "0" * 40,
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual",
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 0}}}
        row_b = {"unit_branch": "feat/rr-fixture", "pre_fix_sha": "f" * 40,
                 "attacks": {"a.py": {"command": "python3 -c \"print('ok')\"", "hash": "a" * 64}},
                 "residual_risk": "fixture residual",
                 "gates": {"python3 ci/scripts/probe.py": {"rc": 1}}}
        files = {
            "ci/lead-gate-required-commands.txt": "python3 ci/scripts/probe.py  # measured ~0.1s\n",
            "docs/rigor/feat_rr-fixture.jsonl": pressure_row + "\n" + block_row + "\n",
            "docs/rigor/feat_rr-fixture.anticipation.jsonl": json.dumps(row_a) + "\n" + json.dumps(row_b) + "\n",
            "docs/README-fixture.md": "line one\n",
            "docs/plans/99-fixture/proposals/contract.md": _VALID_CONTRACT,
        }
        _commit(work, "ci: touch a gate script (no ts on either row)", files)
        r = _run_check_in(work)
        _assert(not r.ok(), "RR18", "two ts-less candidate rows must FAIL loudly, never silently "
                                     "pick one")
        _assert(any("AMBIGUOUS" in f for f in r.failures), "RR18", f"{r.failures}")


RR_FIXTURES = [
    ("RR1", fixture_rr1_not_armed_docs_only),
    ("RR2", fixture_rr2_armed_no_record),
    ("RR3", fixture_rr3_record_no_pressure_row),
    ("RR4", fixture_rr4_no_contract_file),
    ("RR5", fixture_rr5_full_disclosure_allows),
    ("RR6", fixture_rr6_bad_citation_fails),
    ("RR7", fixture_rr7_near_identical_contract_fails),
    ("RR8", fixture_rr8_noop_shapes),
    ("RR9", fixture_rr9_ancestry_advisory_only),
    ("RR10", fixture_rr10_allowlisted_unit_noop),
    ("RR11", fixture_rr11_allowlist_only_shrinks),
    ("RR12a", fixture_rr12a_no_open_block_row_is_a_noop),
    ("RR12b", fixture_rr12b_open_block_no_anticipation_file_fails),
    ("RR12c", fixture_rr12c_shape_allows_despite_unresolvable_pre_fix_sha),
    ("RR12d", fixture_rr12d_real_reproducing_witness_allows),
    ("RR12e", fixture_rr12e_real_nonreproducing_witness_warns_never_fails),
    ("RR12f", fixture_rr12f_grandfathered_unit_no_file_allows),
    ("RR12g", fixture_rr12g_tracked_bash_path_allows),
    ("RR12h", fixture_rr12h_untracked_bash_path_fails_shape),
    ("RR13", fixture_rr13_r12_grandfather_only_shrinks),
    ("RR19", fixture_rr19_required_commands_only_shrinks),
    ("RR14", fixture_rr14_missing_gates_fails),
    ("RR15", fixture_rr15_complete_gates_rc_zero_allows),
    ("RR16", fixture_rr16_nonzero_rc_fails),
    ("RR17", fixture_rr17_interleaved_export_selects_by_ts_not_position),
    ("RR18", fixture_rr18_ambiguous_pool_without_ts_fails_loudly),
]


def self_test() -> int:
    failures: list[str] = []
    for name, fn in RR_FIXTURES:
        try:
            fn()
            print(f"check-rigor-record[{name}]: OK")
        except Failure as e:
            failures.append(f"{name}: {e}")
            print(f"check-rigor-record[{name}]: FAIL — {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: unexpected exception: {e!r}")
            print(f"check-rigor-record[{name}]: FAIL (unexpected exception) — {e!r}", file=sys.stderr)
    if failures:
        print("check-rigor-record: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"check-rigor-record: all {len(RR_FIXTURES)} self-test fixture(s) passed.")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    if "--check-allowlist-only-shrinks" in argv:
        return check_allowlist_only_shrinks()
    if "--check-r12-grandfather-only-shrinks" in argv:
        return check_r12_grandfather_only_shrinks()
    if "--check-required-commands-only-shrinks" in argv:
        return check_required_commands_only_shrinks()
    result = run_check()
    for w in result.warnings:
        print(f"check-rigor-record: WARNING (advisory) — {w}")
    if not result.ok():
        print("check-rigor-record: FAIL", file=sys.stderr)
        for f in result.failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("check-rigor-record: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
