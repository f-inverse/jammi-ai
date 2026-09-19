"""The one ancestry rule shared by every gate that judges whether a
committed artifact's own build-ref is still reachable from HEAD.

`check_cuda_run_artifacts.py` and `check_pod_build_timings.py` both import
`check_ancestry` from here, so the two gates cannot disagree: a branch tip
rewritten before landing (rebased, or merged as a 2-parent merge of a
rebased tip) passes both through the same `merged_as` rescue, or neither.

PROPERTY (the one anchor model): PASS when `git_sha` is an ancestor of
HEAD, OR -- for a branch tip that was rewritten before landing, so
`git_sha` itself can never be an ancestor of anything again -- when
`merged_as` is present, well-typed (40 lowercase hex chars), paired with
`merged_via_pr`, and is ITSELF an ancestor of HEAD. `merged_as` never
REPLACES `git_sha` (the tip that was actually measured/run is kept
verbatim); it only ever supplements it. FAIL names BOTH the `git_sha` and
(when a rescue was attempted) the `merged_as` verdict, never one alone --
a reader debugging a FAIL must never have to re-derive which of the two
was even attempted.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# `shutil.rmtree` during a `tempfile.TemporaryDirectory`'s teardown can hit
# `OSError: [Errno 39] Directory not empty: '.git'` -- a race between tempdir
# cleanup and a background `git maintenance`/`gc --auto` process a throwaway
# repo's `git init`/`add`/`commit`/`clone` call can spawn. `-c gc.auto=0 -c
# gc.autoDetach=false -c maintenance.auto=false` kills the background writer
# AT THE SOURCE for every git invocation this module makes.
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    if cmd and cmd[0] == "git":
        cmd = ["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *cmd[1:]]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def is_shallow_repository(repo_root: Path) -> bool:
    proc = run(["git", "rev-parse", "--is-shallow-repository"], repo_root)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def is_ancestor(sha: str, repo_root: Path, target: str = "HEAD") -> bool:
    proc = run(["git", "merge-base", "--is-ancestor", sha, target], repo_root)
    return proc.returncode == 0


def check_ancestry(data: dict, repo_root: Path, ancestor_message: str) -> list[str]:
    """The ONE anchor model (see module docstring): `merged_as` when
    present, well-typed and an ancestor of HEAD; else `git_sha` when an
    ancestor of HEAD; else fail naming both. `ancestor_message` is the
    CALLER's own prose (the two importing gates cite different docs
    sections for the SAME rule) -- this function owns the DECISION, never
    the wording of why it matters to a given artifact class. `data`
    missing/malformed `git_sha` returns `[]`: the caller's own schema
    rule (`check_schema_types`) already reports that, and this function
    never re-derives a finding its caller already owns."""
    sha = data.get("git_sha")
    if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
        return []
    if is_ancestor(sha, repo_root):
        return []

    merged_as = data.get("merged_as")
    merged_via_pr = data.get("merged_via_pr")
    if isinstance(merged_as, str) and GIT_SHA_RE.match(merged_as) and merged_via_pr is not None:
        if is_ancestor(merged_as, repo_root):
            return []
        return [
            f"git_sha {sha} {ancestor_message} merged_as {merged_as} (PR #{merged_via_pr}) is "
            "ALSO not an ancestor of HEAD — the squash-landing claim does not hold either."
        ]

    return [f"git_sha {sha} {ancestor_message}"]
