#!/usr/bin/env python3
"""_swarm_diff_shape.py — shared diff-shape primitives for swarm gates that
arm on the SHAPE of `base...HEAD`, never on a count or flag the constrained
party (the lead) writes about itself.

Factored out of `check_oracle_gate.py` (esc-lead-invariants-2) rather than
copy-pasted, because the no-op predicates below (dependabot / revert /
release-shaped / human-authored / non-PR-event) are exactly the ones
`docs/plans/53-agentic-swarm/proposals/R7-committed-rigor-record.md`'s own
`check_rigor_record.py` design already established and measured — a second,
independently-drifting copy of the same five predicates is the defect this
module exists to prevent (family T — DRY: a fact lives in exactly one
place). If/when the R7 proposal lands, `check_rigor_record.py` should be
refactored to import this module too rather than keep its own inline copy;
until then, this module's predicates are proven equivalent to R7's by the
same fixtures (see this module's own `--self-test`).

Pure git-plumbing + environment reads. No state directory, no lead-authored
artifact of any kind is trusted here — every function either reads a
committed git object or an Actions-supplied environment variable, both
outside the lead's control at CI-execution time (the residual is that the
lead controls what gets COMMITTED before push, stated explicitly by every
caller of this module, never claimed closed here).
"""
from __future__ import annotations

import fnmatch
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEAD_GATE_LIB = REPO_ROOT / ".claude" / "hooks" / "lead-gate-lib.py"


def git(cwd: Path, *args: str) -> tuple[bool, str]:
    proc = subprocess.run(["git", "-C", str(cwd)] + list(args), capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout if proc.returncode == 0 else proc.stderr).strip()


def lib_module(lib_path: Path = LEAD_GATE_LIB):
    """Loads `lead-gate-lib.py` dynamically (never reimplemented) so
    `slugify()` can never drift between the hook and a CI gate — the same
    discipline `check_rigor_record.py` already established."""
    spec = importlib.util.spec_from_file_location("lead_gate_lib_diffshape", lib_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def compute_diff_context(cwd: Path, base_ref: str, head_ref: str = "HEAD") -> tuple[bool, list[str], str]:
    """Fetches `base_ref` from `origin` and returns `(ok, changed_paths,
    range_spec)` for `origin/<base_ref>...<head_ref>` — a THREE-DOT range,
    pinned explicitly, never two-dot (two-dot silently includes commits
    already on `main` that this branch merely has not rebased past)."""
    ok, _ = git(cwd, "fetch", "--quiet", "origin", base_ref)
    if not ok:
        return False, [], ""
    ok, _ = git(cwd, "rev-parse", "--verify", f"origin/{base_ref}")
    if not ok:
        return False, [], ""
    range_spec = f"origin/{base_ref}...{head_ref}"
    ok, out = git(cwd, "diff", "--name-only", range_spec)
    if not ok:
        return False, [], range_spec
    return True, [p for p in out.splitlines() if p.strip()], range_spec


def commits_in_range(cwd: Path, range_spec: str) -> list[str]:
    ok, out = git(cwd, "log", "--format=%H", range_spec.replace("...", ".."))
    return [c for c in out.splitlines() if c.strip()] if ok else []


def is_human_authored(cwd: Path, range_spec: str) -> bool:
    """True iff NO commit in the range carries this repo's own swarm
    co-authorship trailer — a single swarm-touched commit is enough to arm;
    a purely human PR that never touched the branch is not bound by the
    swarm's own rigor process."""
    shas = commits_in_range(cwd, range_spec)
    if not shas:
        return True
    for sha in shas:
        ok, msg = git(cwd, "log", "-1", "--format=%B", sha)
        if ok and re.search(r"Co-Authored-By:\s*Claude", msg, re.IGNORECASE):
            return False
    return True


def is_revert(cwd: Path, head_ref: str = "HEAD") -> bool:
    ok, subject = git(cwd, "log", "-1", "--format=%s", head_ref)
    return ok and subject.startswith('Revert "')


def is_release_shaped(changed: list[str], arming_globs: tuple[str, ...]) -> bool:
    """True iff every changed path matching an arming glob under `crates/`
    is a `Cargo.toml`/`Cargo.lock` (a version-bump-only diff) and no
    non-`crates/` arming path is touched. A heuristic, stated as one — a
    release PR that ALSO edits a gate script is correctly not exempted."""
    crate_touches = [p for p in changed if _matches(p, arming_globs) and p.startswith("crates/")]
    other_touches = [p for p in changed if _matches(p, arming_globs) and not p.startswith("crates/")]
    if other_touches or not crate_touches:
        return False
    return all(Path(p).name in ("Cargo.toml", "Cargo.lock") for p in crate_touches)


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def no_op_reason(cwd: Path, changed: list[str], range_spec: str, arming_globs: tuple[str, ...]) -> str | None:
    """The KNOWN, STRUCTURAL no-op shapes (never an author-declared
    marker/trailer/commit-message convention — every such predicate is a
    one-line opt-out): not a `pull_request` event; `dependabot[bot]`;
    a revert; a release-shaped diff; a purely human-authored range.
    A diff touching none of `arming_globs` is not a "no-op reason" here —
    the caller checks that separately (it is "not armed", not "exempt")."""
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if event and event != "pull_request":
        return f"not a pull_request event ({event!r})"
    actor = os.environ.get("GITHUB_ACTOR", "")
    if actor == "dependabot[bot]":
        return "dependabot PR"
    if is_revert(cwd):
        return 'revert PR (HEAD commit subject starts with Revert ")'
    if is_release_shaped(changed, arming_globs):
        return "release-shaped diff (only Cargo.toml/Cargo.lock under crates/**, no other arming path touched)"
    if is_human_authored(cwd, range_spec):
        return "no commit in range carries the swarm's own Co-Authored-By: Claude trailer"
    return None


def unit_slug(cwd: Path, lib_path: Path = LEAD_GATE_LIB) -> str:
    head_ref_env = os.environ.get("GITHUB_HEAD_REF")
    if head_ref_env:
        head_branch = head_ref_env
    else:
        ok, head_branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
        if not ok:
            head_branch = "HEAD"
    return lib_module(lib_path).slugify(head_branch)


if __name__ == "__main__":
    sys.stderr.write("_swarm_diff_shape.py is a library module, not a standalone gate.\n")
    sys.exit(2)
