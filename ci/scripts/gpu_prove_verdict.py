#!/usr/bin/env python3
"""GPU-prove verdict consumer (esc-084, issue #454; check-once/fail-loud,
operator direction 2026-09-03).

**Guarded property**: proof surface == shipped surface, proven ONCE per
commit and SHARED. Every release-publishing workflow (CUDA and non-CUDA
alike — server image, release binaries, cu12 wheel, crates.io, npm, every
PyPI dist) calls the `_gpu-proof-required.yml` reusable, which runs this
script instead of renting hardware itself. `gpu-prove.yml` is never
triggered from here -- see that workflow's own header for the canonical
statement of why; a publisher gates on the SUMMARY of a prove execution
already recorded against the commit it is promoting, never on a fresh
measurement it starts.

## The rule (esc-084 control b, most-recent-measurement-wins, CHECK ONCE)

For each required arch (the shipped `GENCODE_ARCHES` silicon axis,
`check_gpu_parity_matrix.py`'s own parser — never a hand-typed list): a
"measurement" is a COMPLETED (non-`skipped`) conclusion of the job named
exactly `GPU prove on RunPod (<arch>)` (`JOB_NAME_TEMPLATE` below — the ONE
string `check_gpu_prove_once.py`'s P4 rule pins against `gpu-prove.yml`'s
own matrix `name:` line), taken from the LATEST ATTEMPT of that job
(`filter=latest`, so a `gh run rerun --failed` supersedes a stale attempt in
place) on a run of `gpu-prove.yml` whose `head_sha` is the commit this
caller promotes. `skipped` is not a measurement (e.g. a labeled-event run
for a different label never touched this arch). A run that has not yet
completed contributes NO measurement for any arch — it is simply invisible
to this check, never a reason to wait: this script checks ONCE and returns.

When an arch's LATEST ATTEMPT in a given run is itself still in progress
(no `completed_at` yet), `filter=latest` would otherwise hide an EARLIER,
already-completed attempt of that same run entirely — including a red one
mid-rerun. `collect_measurements` closes this: it falls back to that one
run's own most recent COMPLETED attempt for the arch (a `filter=all`
re-query, lazy and cached per run), and a red completed attempt there still
denies; only a run with no completed attempt at all for the arch
contributes no measurement. This is still never a wait — the fallback reads
what has already completed, once, and returns.

Recency: the MOST RECENT measurement per arch is the one with
the greatest job `completed_at` — never run id alone, because
`gh run rerun <id> --failed` re-runs an EXISTING (possibly numerically
older) run id with a fresh completed_at, and a numerically newer run id can
finish before an older one is rerun. Ties break on run id (higher wins). A
later red measurement REVOKES an earlier green until a re-run succeeds
(fail-closed).

Check-once, fail-loud (operator direction, 2026-09-03 — supersedes the
earlier polling design; nothing here ever starts a prove run, and nothing
here ever waits for one):
  1. If every required arch's most recent COMPLETED measurement is
     `success`, exit 0 immediately.
  2. Else DENY immediately (no poll, no deadline, no grace window — an
     in-progress run at this sha is not a measurement and never delays or
     satisfies this check): every arch with no completed measurement at
     all, or whose most recent completed measurement is not `success`, is
     named in the error, each with its own remedy — `gh workflow run
     gpu-prove.yml --ref <sha-or-tag>` for a missing measurement, `gh run
     rerun <run_id> --failed` for a red one (re-running one failed leg is
     one GPU pod, not a fresh whole-workflow dispatch) — then re-run this
     workflow's failed jobs once every arch is green. Prove first, then
     tag: a tag push on a commit whose prove is not already green fails
     every release workflow immediately, by design.

Any HTTP/JSON error is a hard DENY (exit 1) — never vacuous: a network
hiccup must never read as "nothing to prove" and let something silently
promote.

`fetch(url, token) -> dict` is injected so `test_gpu_prove_verdict.py` can
drive every branch above with a fake API, no network.

Run (real use, inside `_gpu-proof-required.yml`):
  GITHUB_TOKEN=<token> python3 ci/scripts/gpu_prove_verdict.py \\
      --repo owner/repo --sha <sha>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import check_gpu_parity_matrix as gpu_parity_matrix  # noqa: E402

API_BASE = "https://api.github.com"
DEFAULT_WORKFLOW = "gpu-prove.yml"
# The ONE job-name template this consumer matches -- pinned against
# `gpu-prove.yml`'s own matrix `name:` line by `check_gpu_prove_once.py`'s
# P4 rule so the producer and this consumer can never independently drift.
JOB_NAME_TEMPLATE = "GPU prove on RunPod ({arch})"
PER_PAGE = 100
SUCCESS = "success"  # exact string equality only -- never a substring/prefix check.


class VerdictError(Exception):
    """Any HTTP/JSON error talking to the GitHub API -- always a hard deny,
    never a silent/vacuous pass."""


@dataclass(frozen=True)
class Measurement:
    arch: str
    run_id: int
    job_id: int
    conclusion: str | None
    completed_at: str | None
    html_url: str


FetchFn = Callable[[str, str], dict]


def default_fetch(url: str, token: str) -> dict:
    """Real GitHub REST call. Never used by the test suite (which injects
    its own `fetch`); any transport/HTTP/JSON failure raises `VerdictError`."""
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
            body = resp.read()
    except (urllib.error.URLError, OSError) as e:
        raise VerdictError(f"GET {url} failed: {e}") from e
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise VerdictError(f"GET {url} returned invalid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise VerdictError(f"GET {url} returned a non-object JSON body")
    return parsed


def _paginated(fetch: FetchFn, token: str, url: str, key: str) -> list[dict]:
    """GETs `url` with an incrementing `page=` parameter while the page is
    full (`PER_PAGE` items) -- the plain page-number equivalent of following
    a `Link: rel="next"` header, needing no header access from `fetch`
    (which returns parsed JSON only, per the injected-callable contract).
    Stops the moment a page returns fewer than `PER_PAGE` items (including
    zero)."""
    out: list[dict] = []
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        page_url = f"{url}{sep}per_page={PER_PAGE}&page={page}"
        try:
            data = fetch(page_url, token)
        except VerdictError:
            raise
        except Exception as e:  # noqa: BLE001 - any fetch failure is a hard deny
            raise VerdictError(f"GET {page_url} failed: {e}") from e
        if not isinstance(data, dict) or key not in data:
            raise VerdictError(f"GET {page_url} returned an unexpected shape (missing {key!r})")
        items = data[key]
        if not isinstance(items, list):
            raise VerdictError(f"GET {page_url} field {key!r} is not a list")
        out.extend(items)
        if len(items) < PER_PAGE:
            break
        page += 1
    return out


def list_runs(fetch: FetchFn, token: str, repo: str, workflow: str, sha: str) -> list[dict]:
    """Every run of `workflow` at `head_sha == sha`, all pages, explicitly
    sorted by run id descending -- never trusting API order.

    Defense in depth (esc-084 control b): the `head_sha`/workflow-path
    query params ask the SERVER to scope the result, but this never TRUSTS
    that scoping blindly -- a returned run whose own `head_sha` disagrees
    with what was asked for, or whose own `path` names a different workflow
    file, is dropped client-side too, so a proxy/cache bug or a malformed
    fixture can never smuggle a foreign commit's or a foreign workflow's
    green measurement into this commit's verdict.

    Advisory A5 fix: `path` must be PRESENT and equal the exact repo-
    relative path `.github/workflows/<workflow>` -- never `"path" not in r`
    (a record with no `path` key at all is REFUSED, not vacuously
    accepted) and never a bare `endswith` (which a nested path like
    `vendor/.github/workflows/<workflow>` would also satisfy)."""
    url = f"{API_BASE}/repos/{repo}/actions/workflows/{workflow}/runs?head_sha={sha}"
    runs = _paginated(fetch, token, url, "workflow_runs")
    want_path = f".github/workflows/{workflow}"
    runs = [r for r in runs if r.get("head_sha") == sha and r.get("path") == want_path]
    runs.sort(key=lambda r: r.get("id", 0), reverse=True)
    return runs


def list_jobs(fetch: FetchFn, token: str, repo: str, run_id: int, filter_mode: str = "latest") -> list[dict]:
    """`filter=latest` (the default) -- the latest ATTEMPT of each job in the
    run, so a `gh run rerun <id> --failed` supersedes a stale attempt in
    place rather than leaving a first-attempt failure to be read as a fresh
    one. `filter_mode="all"` is used by `collect_measurements`'s F5 fallback
    below -- ONLY when the latest attempt is itself still in progress -- to
    recover that run's own most recent COMPLETED attempt for one arch,
    never to second-guess a completed latest attempt."""
    url = f"{API_BASE}/repos/{repo}/actions/runs/{run_id}/jobs?filter={filter_mode}"
    return _paginated(fetch, token, url, "jobs")


def collect_measurements(
    fetch: FetchFn,
    token: str,
    repo: str,
    runs: list[dict],
    required_arches: list[str],
) -> dict[str, list[Measurement]]:
    """Returns `{arch: [Measurement, ...]}`.

    A job is a measurement for its arch iff its name matches
    `JOB_NAME_TEMPLATE` for a required arch, its conclusion is present and
    not `"skipped"`, and it has a `completed_at` (a job with no
    `completed_at` is still running -- this is a check-once, fail-loud
    consumer, so an incomplete job simply contributes no measurement; it is
    never a reason to wait, same as a run whose own `status` is not yet
    `"completed"`).

    F5 audit fix (fail-open window): `filter=latest` returns ONLY the latest
    attempt of each job. When that latest attempt is itself still running
    (`completed_at` is `None`), `filter=latest` hides any EARLIER,
    already-COMPLETED attempt of the same arch's job in this SAME run
    entirely -- including a red one -- so without this fallback an older
    run's stale green could read as "most recent completed" while a newer,
    contradicting red attempt sits invisible mid-rerun. When the latest
    attempt is running, this run is re-queried with `filter=all` (once,
    lazily, only for runs that actually need it) and the run's OWN most
    recent COMPLETED (non-`skipped`) attempt for that arch is used instead
    -- a red one there still denies. Only when the run has NO completed
    attempt at all for that arch (e.g. its first and only attempt is still
    in progress) does the run contribute no measurement for it -- never a
    wait, never a poll."""
    by_arch: dict[str, list[Measurement]] = {a: [] for a in required_arches}
    name_to_arch = {JOB_NAME_TEMPLATE.format(arch=a): a for a in required_arches}
    for run in runs:
        run_id = run.get("id")
        latest_jobs = list_jobs(fetch, token, repo, run_id, filter_mode="latest")
        all_jobs_cache: list[dict] | None = None
        for job in latest_jobs:
            arch = name_to_arch.get(job.get("name"))
            if arch is None:
                continue
            conclusion = job.get("conclusion")
            completed_at = job.get("completed_at")
            job_id = job.get("id")
            html_url = job.get("html_url") or ""
            if completed_at is None:
                # F5 fallback: the latest attempt is still in flight -- fetch
                # (and cache, once per run) the unfiltered attempt list, and
                # use THIS run's own most recent completed attempt for this
                # arch instead. A run whose fallback also finds nothing
                # completed contributes no measurement, exactly like today.
                if all_jobs_cache is None:
                    all_jobs_cache = list_jobs(fetch, token, repo, run_id, filter_mode="all")
                completed_attempts = [
                    j
                    for j in all_jobs_cache
                    if name_to_arch.get(j.get("name")) == arch
                    and j.get("conclusion") != "skipped"
                    and j.get("completed_at") is not None
                ]
                if not completed_attempts:
                    continue  # no completed attempt in this run for this arch -- no measurement, never a wait
                job = max(completed_attempts, key=lambda j: (j.get("completed_at") or "", j.get("id", 0)))
                conclusion = job.get("conclusion")
                completed_at = job.get("completed_at")
                job_id = job.get("id")
                html_url = job.get("html_url") or ""
            if conclusion == "skipped":
                continue  # no measurement was made (rule F)
            by_arch[arch].append(
                Measurement(
                    arch=arch,
                    run_id=run_id,
                    job_id=job_id,
                    conclusion=conclusion,
                    completed_at=completed_at,
                    html_url=html_url,
                )
            )
    return by_arch


def most_recent(measurements: list[Measurement]) -> Measurement | None:
    """Greatest `completed_at`, run id as the tiebreak (a `gh run rerun
    <id> --failed` re-runs an EXISTING, possibly numerically older, run id
    with a fresh `completed_at`, so recency by run id alone would be wrong).
    ISO8601 UTC (`Z`) timestamps compare correctly as plain strings."""
    if not measurements:
        return None
    return max(measurements, key=lambda m: (m.completed_at or "", m.run_id))


@dataclass(frozen=True)
class Verdict:
    ok: bool
    proofs: dict[str, Measurement]
    missing: list[str]
    failing: dict[str, Measurement]


def evaluate(by_arch: dict[str, list[Measurement]], required_arches: list[str]) -> Verdict:
    # Advisory A6 fix -- an arity floor: `not missing and not failing` is
    # vacuously `True` when `required_arches` is empty (`not [] and not
    # {}`), which would report a verdict as PROVEN for a caller that asked
    # about nothing. A caller with zero required arches asked a malformed
    # question; refuse rather than silently promote it as satisfied.
    if not required_arches:
        raise VerdictError("evaluate() called with zero required arches -- refusing a vacuous verdict")
    proofs: dict[str, Measurement] = {}
    missing: list[str] = []
    failing: dict[str, Measurement] = {}
    for arch in sorted(required_arches):
        best = most_recent(by_arch.get(arch, []))
        if best is None:
            missing.append(arch)
            continue
        if best.conclusion == SUCCESS:
            proofs[arch] = best
        else:
            failing[arch] = best
    return Verdict(ok=not missing and not failing, proofs=proofs, missing=missing, failing=failing)


def check_once(
    fetch: FetchFn,
    token: str,
    repo: str,
    workflow: str,
    sha: str,
    required_arches: list[str],
) -> Verdict:
    runs = list_runs(fetch, token, repo, workflow, sha)
    by_arch = collect_measurements(fetch, token, repo, runs, required_arches)
    return evaluate(by_arch, required_arches)


def required_arches() -> list[str]:
    return sorted(gpu_parity_matrix.load_shipped_cuda_silicon())


def run(
    *,
    repo: str,
    sha: str,
    workflow: str,
    fetch: FetchFn,
    token: str,
    arches: list[str],
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    """Check-once, fail-loud (operator direction, 2026-09-03): exactly one
    lookup, no poll, no deadline, no grace window. An in-progress run at
    this sha is invisible to this check -- it is never a measurement and
    never delays or satisfies the verdict."""
    try:
        verdict = check_once(fetch, token, repo, workflow, sha, arches)
    except VerdictError as e:
        print(f"::error::gpu-prove-verdict: {e}", file=err)
        return 1

    if verdict.ok:
        for arch in sorted(verdict.proofs):
            m = verdict.proofs[arch]
            print(f"PROVEN {arch}: run={m.run_id} job={m.job_id} {m.html_url}", file=out)
        return 0

    if verdict.missing:
        print(
            f"::error::gpu-prove-verdict: DENY -- no GPU-prove measurement for "
            f"{', '.join(verdict.missing)} at {sha} -- dispatch the prove lane: "
            f"gh workflow run {workflow} --ref <tag-or-branch-to-prove>, wait for green, "
            f"then re-run this workflow's failed jobs. Prove first, then tag.",
            file=err,
        )
    for arch in sorted(verdict.failing):
        m = verdict.failing[arch]
        print(
            f"::error::gpu-prove-verdict: DENY -- {arch}'s most recent measurement is "
            f"{m.conclusion!r} (run={m.run_id} job={m.job_id} {m.html_url}) -- "
            f"gh run rerun {m.run_id} --failed",
            file=err,
        )
    return 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--sha", required=True, help="The exact commit this caller is promoting.")
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    return ap


def main(argv: list[str]) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::error::gpu-prove-verdict: GITHUB_TOKEN is not set", file=sys.stderr)
        return 1
    return run(
        repo=args.repo,
        sha=args.sha,
        workflow=args.workflow,
        fetch=default_fetch,
        token=token,
        arches=required_arches(),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
