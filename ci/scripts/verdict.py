#!/usr/bin/env python3
"""A tree's recorded verdict: check once, fail loud.

A run of a proving workflow records the tree it tests — the
`record-subject` action uploads an artifact named `subject-<tree>` at the run's
start — so a verdict over tree T reads every run bound to T, takes each
required job's measurements from those runs, and holds when every required
job's most recent measurement is a success. A tree is what a run proves, never
a commit: a pull request's run tests its merge commit, the merge lands on `main`
as another commit with the same tree, and a release tags a third.

Two questions are asked of it:

    require --repo R --tree T   The release gate (`_proof-required.yml`): T's GPU
                                prove (`gpu-prove.yml`, one job per shipped CUDA
                                arch), T's CI (`ci.yml`'s summary job, the merge
                                gate), T's build (`build.yml`'s summary job, the
                                artifacts and their consumers) and T's cookbook
                                (`cookbook-gpu.yml`, every recipe and page on
                                real models) all hold. A missing or red
                                measurement DENIES, each naming its remedy.
    probe --lane ci|build       Whether T's lane already holds, as
          --repo R --tree T     `proven=true|false` for `$GITHUB_OUTPUT`, with a
                                notice naming the run whose summary is that
                                measurement. The lane's `plan` skips its test
                                executions on a proven T. An unreadable answer is
                                `proven=false` with a warning, so a run measures
                                T again.

## The rule (most-recent-measurement-wins, check once)

A measurement of a required job is a COMPLETED, non-`skipped` conclusion of
the job named exactly so, taken from the LATEST ATTEMPT of that job
(`filter=latest`, so a `gh run rerun --failed` supersedes a stale attempt in
place) in a run of the requirement's workflow bound to T. `skipped` is not a
measurement: a run that found T already proven skips, and measured nothing. A
job that has not completed contributes no measurement — it is never a reason
to wait.

When a job's latest attempt is still in progress, `filter=latest` would hide an
earlier, already-completed attempt of the same run, including a red one; that
run's own most recent completed attempt is read instead (a `filter=all`
re-query, lazy and cached per run). Only a run with no completed attempt of
the job contributes nothing.

The most recent measurement is the one with the greatest `completed_at`, run
id breaking ties: a rerun gives an existing, possibly numerically older, run a
fresh `completed_at`. A later red measurement revokes an earlier green until a
rerun succeeds.

Any HTTP/JSON error DENIES: an unreadable record never reads as a proof.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402
import github_api  # noqa: E402
from github_api import API_BASE, ApiError, FetchFn  # noqa: E402

SUBJECT_ARTIFACT = "subject-{tree}"
SUCCESS = "success"  # exact string equality only -- never a substring/prefix check.

CI_WORKFLOW = "ci.yml"
# The one job whose success holds every other `ci.yml` job's, pinned against
# `ci.yml`'s `ci-summary` call of `_summary.yml` by `check_gpu_prove_once.py`'s
# P4 rule.
CI_SUMMARY_JOB = "ci-summary / assert"
BUILD_WORKFLOW = "build.yml"
# The one job whose success holds every artifact and every consumer of them,
# pinned the same way against `build.yml`'s `build-summary` call.
BUILD_SUMMARY_JOB = "build-summary / assert"
GPU_WORKFLOW = "gpu-prove.yml"
# Pinned against `gpu-prove.yml`'s own matrix `name:` line by
# `check_gpu_prove_once.py`'s P4 rule, so producer and consumer never drift.
GPU_JOB_TEMPLATE = "GPU prove on RunPod ({arch})"
COOKBOOK_WORKFLOW = "cookbook-gpu.yml"
# The one job that runs every recipe and page and assembles the book, pinned
# against `cookbook-gpu.yml`'s job `name:` by the same P4 rule.
COOKBOOK_JOB = "Cookbook on RunPod"


@dataclass(frozen=True)
class Requirement:
    """What must hold of a tree: `jobs`, measured by runs of `workflow`;
    `remedy` says how a tree with no measurement gets one."""

    workflow: str
    jobs: tuple[str, ...]
    remedy: str


@dataclass(frozen=True)
class Measurement:
    job: str
    run_id: int
    job_id: int
    conclusion: str | None
    completed_at: str | None
    html_url: str


@dataclass(frozen=True)
class Verdict:
    requirement: Requirement
    ok: bool
    proofs: dict[str, Measurement]
    missing: list[str]
    failing: dict[str, Measurement]


def ci_requirement() -> Requirement:
    return Requirement(
        CI_WORKFLOW,
        (CI_SUMMARY_JOB,),
        f"no {CI_WORKFLOW} run has measured this tree: a pull request or a push to main runs it",
    )


def build_requirement() -> Requirement:
    return Requirement(
        BUILD_WORKFLOW,
        (BUILD_SUMMARY_JOB,),
        f"no {BUILD_WORKFLOW} run has measured this tree: a pull request or a push to main runs it",
    )


# The lanes a plan probes, by the name its workflow passes.
PROBE_LANES = {"ci": ci_requirement, "build": build_requirement}


def gpu_requirement() -> Requirement:
    arches = sorted(gpu_parity_matrix.load_shipped_cuda_silicon())
    return Requirement(
        GPU_WORKFLOW,
        tuple(GPU_JOB_TEMPLATE.format(arch=a) for a in arches),
        f"dispatch the prove lane: gh workflow run {GPU_WORKFLOW} --ref <tag-or-branch-holding-the-tree>, "
        "wait for green, then re-run this workflow's failed jobs. Prove first, then tag.",
    )


def cookbook_requirement() -> Requirement:
    return Requirement(
        COOKBOOK_WORKFLOW,
        (COOKBOOK_JOB,),
        f"run the cookbook lane: gh workflow run {COOKBOOK_WORKFLOW} --ref <branch-holding-the-tree> "
        "(or label the pull request `cookbook-gpu`), wait for green, then re-run this workflow's "
        "failed jobs.",
    )


def release_requirements() -> list[Requirement]:
    """What a release requires of its tree: the GPU prove, CI, the build and
    the cookbook."""
    return [gpu_requirement(), ci_requirement(), build_requirement(), cookbook_requirement()]


def bound_runs(fetch: FetchFn, token: str, repo: str, workflow: str, tree: str) -> list[dict]:
    """Every run of `workflow` that recorded `tree` as its subject, newest id
    first. The artifact binds a run to a tree and nothing more: the run's own
    record (read from the API, never from the artifact) must name exactly
    `.github/workflows/<workflow>` — a missing `path`, or a nested one, is
    refused — so another workflow's artifact of the same name is never read
    as this workflow's proof."""
    name = SUBJECT_ARTIFACT.format(tree=tree)
    url = f"{API_BASE}/repos/{repo}/actions/artifacts?name={name}"
    artifacts = github_api.paginated(fetch, token, url, "artifacts")
    run_ids = sorted(
        {
            a["workflow_run"]["id"]
            for a in artifacts
            if a.get("name") == name
            and isinstance(a.get("workflow_run"), dict)
            and isinstance(a["workflow_run"].get("id"), int)
        },
        reverse=True,
    )
    want_path = f".github/workflows/{workflow}"
    runs = [github_api.get(fetch, token, f"{API_BASE}/repos/{repo}/actions/runs/{rid}") for rid in run_ids]
    return [r for r in runs if r.get("path") == want_path]


def collect_measurements(
    fetch: FetchFn, token: str, repo: str, runs: list[dict], jobs: tuple[str, ...]
) -> dict[str, list[Measurement]]:
    """`{job: [Measurement, ...]}` over `runs` (the rule in this module's doc)."""
    by_job: dict[str, list[Measurement]] = {j: [] for j in jobs}
    for run in runs:
        run_id = run.get("id")
        all_attempts: list[dict] | None = None
        for job in github_api.list_jobs(fetch, token, repo, run_id, filter_mode="latest"):
            name = job.get("name")
            if name not in by_job:
                continue
            if job.get("completed_at") is None:
                if all_attempts is None:
                    all_attempts = github_api.list_jobs(fetch, token, repo, run_id, filter_mode="all")
                completed = [
                    j
                    for j in all_attempts
                    if j.get("name") == name
                    and j.get("conclusion") != "skipped"
                    and j.get("completed_at") is not None
                ]
                if not completed:
                    continue
                job = max(completed, key=lambda j: (j.get("completed_at") or "", j.get("id", 0)))
            if job.get("conclusion") == "skipped":
                continue
            by_job[name].append(
                Measurement(
                    job=name,
                    run_id=run_id,
                    job_id=job.get("id"),
                    conclusion=job.get("conclusion"),
                    completed_at=job.get("completed_at"),
                    html_url=job.get("html_url") or "",
                )
            )
    return by_job


def most_recent(measurements: list[Measurement]) -> Measurement | None:
    """Greatest `completed_at`, run id as the tiebreak. ISO8601 UTC (`Z`)
    timestamps compare correctly as plain strings."""
    if not measurements:
        return None
    return max(measurements, key=lambda m: (m.completed_at or "", m.run_id))


def evaluate(requirement: Requirement, by_job: dict[str, list[Measurement]]) -> Verdict:
    # `not missing and not failing` is vacuously true of zero jobs; a
    # requirement that names nothing is a malformed question, never a proof.
    if not requirement.jobs:
        raise ApiError(f"a requirement of {requirement.workflow} names no job -- refusing a vacuous verdict")
    proofs: dict[str, Measurement] = {}
    missing: list[str] = []
    failing: dict[str, Measurement] = {}
    for job in sorted(requirement.jobs):
        best = most_recent(by_job.get(job, []))
        if best is None:
            missing.append(job)
        elif best.conclusion == SUCCESS:
            proofs[job] = best
        else:
            failing[job] = best
    return Verdict(requirement, not missing and not failing, proofs, missing, failing)


def check_once(fetch: FetchFn, token: str, repo: str, requirement: Requirement, tree: str) -> Verdict:
    runs = bound_runs(fetch, token, repo, requirement.workflow, tree)
    return evaluate(requirement, collect_measurements(fetch, token, repo, runs, requirement.jobs))


def require(
    *,
    repo: str,
    tree: str,
    requirements: list[Requirement],
    fetch: FetchFn,
    token: str,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    """The release gate: every requirement holds of `tree`, checked once."""
    denied = False
    for requirement in requirements:
        try:
            verdict = check_once(fetch, token, repo, requirement, tree)
        except ApiError as e:
            print(f"::error::verdict: {requirement.workflow}: {e}", file=err)
            denied = True
            continue
        for job in sorted(verdict.proofs):
            m = verdict.proofs[job]
            print(f"PROVEN {job}: run={m.run_id} job={m.job_id} {m.html_url}", file=out)
        if verdict.missing:
            print(
                f"::error::verdict: DENY -- no measurement of {', '.join(verdict.missing)} for tree "
                f"{tree} -- {requirement.remedy}",
                file=err,
            )
        for job in sorted(verdict.failing):
            m = verdict.failing[job]
            print(
                f"::error::verdict: DENY -- {job}'s most recent measurement is {m.conclusion!r} "
                f"(run={m.run_id} job={m.job_id} {m.html_url}) -- gh run rerun {m.run_id} --failed",
                file=err,
            )
        denied = denied or not verdict.ok
    return 1 if denied else 0


def probe(
    *, repo: str, tree: str, lane: str, fetch: FetchFn, token: str, out=sys.stdout, err=sys.stderr
) -> int:
    """`proven=true` when `tree`'s `lane` already holds, `proven=false`
    otherwise. Never fails: an unreadable record means the run measures
    again."""
    requirement = PROBE_LANES[lane]()
    try:
        verdict = check_once(fetch, token, repo, requirement, tree)
    except ApiError as e:
        print(f"::warning::verdict: {e} -- this run measures tree {tree} again", file=err)
        print("proven=false", file=out)
        return 0
    print(f"proven={'true' if verdict.ok else 'false'}", file=out)
    if verdict.ok:
        (job,) = requirement.jobs
        m = verdict.proofs[job]
        print(f"::notice::tree {tree} is proven by run {m.run_id} ({m.html_url})", file=err)
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", choices=["require", "probe"])
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--tree", required=True, help="The git tree id the verdict is about.")
    ap.add_argument("--lane", choices=sorted(PROBE_LANES), help="probe: the lane whose plan asks.")
    args = ap.parse_args(argv)
    if (args.question == "probe") != (args.lane is not None):
        ap.error("--lane is for probe, and probe needs it")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::error::verdict: GITHUB_TOKEN is not set", file=sys.stderr)
        return 1
    if args.question == "probe":
        return probe(repo=args.repo, tree=args.tree, lane=args.lane, fetch=github_api.default_fetch, token=token)
    return require(
        repo=args.repo,
        tree=args.tree,
        requirements=release_requirements(),
        fetch=github_api.default_fetch,
        token=token,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
