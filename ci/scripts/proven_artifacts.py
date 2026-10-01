#!/usr/bin/env python3
"""The artifacts a release promotes: those of the run that proved its tree.

A release compiles nothing. `ci.yml` builds every artifact a tree ships once,
through the reusable workflow that defines it, and every check of that run
that needs one installs that build. The run whose `ci-summary / assert` is the
tree's most recent green measurement — `verdict.py`'s rule, the one the release
gate requires — therefore holds the bytes its checks exercised, and a
publisher downloads them by that run's id.

    proven_artifacts.py --repo R --tree T --artifacts "A B ..."

prints `run=<id>` for `$GITHUB_OUTPUT` when T is proven and that run still
holds every artifact named. It DENIES (exit 1), naming the remedy, when T has
no green `ci.yml` run, or when that run lacks a named artifact: never uploaded,
or expired. A run's artifacts and its `subject-<tree>` binding share one
retention, so a tree older than that is measured again before it can be
released.

Any HTTP/JSON error DENIES: an unreadable record is never a release's source.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import github_api  # noqa: E402
import verdict  # noqa: E402
from github_api import API_BASE, ApiError, FetchFn  # noqa: E402

REMEDY = (
    "measure the tree again: a ci.yml run over it (a pull request or a push to main holding it) "
    "builds its artifacts"
)


def run_artifacts(fetch: FetchFn, token: str, repo: str, run_id: int) -> dict[str, dict]:
    """`{name: artifact}` for every artifact `run_id` uploaded."""
    url = f"{API_BASE}/repos/{repo}/actions/runs/{run_id}/artifacts"
    return {
        a["name"]: a
        for a in github_api.paginated(fetch, token, url, "artifacts")
        if isinstance(a.get("name"), str)
    }


def resolve(
    *,
    repo: str,
    tree: str,
    names: list[str],
    fetch: FetchFn,
    token: str,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    if not names:
        print("::error::proven_artifacts: no artifact named -- refusing a vacuous release", file=err)
        return 1
    try:
        found = verdict.check_once(fetch, token, repo, verdict.ci_requirement(), tree)
    except ApiError as e:
        print(f"::error::proven_artifacts: {e}", file=err)
        return 1
    if not found.ok:
        for m in found.failing.values():
            print(
                f"::error::proven_artifacts: DENY -- tree {tree}'s most recent {verdict.CI_WORKFLOW} "
                f"measurement is {m.conclusion!r} (run={m.run_id} {m.html_url})",
                file=err,
            )
        if found.missing:
            print(f"::error::proven_artifacts: DENY -- tree {tree} has no {verdict.CI_WORKFLOW} run -- {REMEDY}", file=err)
        return 1
    proof = found.proofs[verdict.CI_SUMMARY_JOB]
    try:
        held = run_artifacts(fetch, token, repo, proof.run_id)
    except ApiError as e:
        print(f"::error::proven_artifacts: {e}", file=err)
        return 1
    absent = [n for n in names if n not in held]
    expired = [n for n in names if n in held and held[n].get("expired")]
    if absent or expired:
        for name in absent:
            print(f"::error::proven_artifacts: DENY -- run {proof.run_id} uploaded no artifact {name!r}", file=err)
        for name in expired:
            print(f"::error::proven_artifacts: DENY -- run {proof.run_id}'s artifact {name!r} has expired -- {REMEDY}", file=err)
        return 1
    print(f"::notice::tree {tree}'s artifacts come from run {proof.run_id} ({proof.html_url})", file=err)
    print(f"run={proof.run_id}", file=out)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--tree", required=True, help="The git tree id being released.")
    ap.add_argument("--artifacts", required=True, help="Space-separated artifact names the release promotes.")
    args = ap.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::error::proven_artifacts: GITHUB_TOKEN is not set", file=sys.stderr)
        return 1
    return resolve(
        repo=args.repo,
        tree=args.tree,
        names=args.artifacts.split(),
        fetch=github_api.default_fetch,
        token=token,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
