#!/usr/bin/env python3
"""GPU-prove verdict consumer (esc-084, issue #454).

**Guarded property**: proof surface == shipped surface, proven ONCE per
commit and SHARED. Every CUDA release lane (server image, release binaries,
cu12 wheel) calls the `_gpu-proof-required.yml` reusable, which runs this
script instead of renting hardware itself. `gpu-prove.yml` is never
triggered from here -- see that workflow's own header for the canonical
statement of why; a publisher gates on the SUMMARY of a prove execution
already recorded against the commit it is promoting, never on a fresh
measurement it starts.

## The rule (esc-084 control b, most-recent-measurement-wins)

For each required arch (the shipped `GENCODE_ARCHES` silicon axis,
`check_gpu_parity_matrix.py`'s own parser — never a hand-typed list): a
"measurement" is a completed (non-`skipped`) conclusion of the job named
exactly `GPU prove on RunPod (<arch>)` (`JOB_NAME_TEMPLATE` below — the ONE
string `check_gpu_prove_once.py`'s P4 rule pins against `gpu-prove.yml`'s
own matrix `name:` line), taken from the LATEST ATTEMPT of that job
(`filter=latest`, so a `gh run rerun --failed` supersedes a stale attempt in
place) on a run of `gpu-prove.yml` whose `head_sha` is the commit this
caller promotes. `skipped` is not a measurement (e.g. a labeled-event run
for a different label never touched this arch).

Recency: the MOST RECENT measurement per arch is the one with
the greatest job `completed_at` — never run id alone, because
`gh run rerun <id> --failed` re-runs an EXISTING (possibly numerically
older) run id with a fresh completed_at, and a numerically newer run id can
finish before an older one is rerun. Ties break on run id (higher wins). A
later red measurement REVOKES an earlier green until a re-run succeeds
(fail-closed).

Wait state (operator direction, 2026-09-03 — no appearance grace; nothing
here ever starts a prove run):
  1. If every required arch's most recent measurement is `success`, exit 0
     immediately, even if a newer run is queued/in progress at this sha (a
     publisher that already started consumes the verdict as of its own
     start — this bounds the wait).
  2. Else, while any candidate run at this sha is queued/in_progress, poll
     (`--poll-seconds`) until `--deadline-minutes` elapses. This serves the
     tag-then-dispatch overlap (Z): a dev's dispatch that began just before
     the publisher's first poll is still picked up.
  3. Else, if no measurement exists for some arch at all, DENY immediately
     (no grace — nothing auto-starts the prove; the remedy is a dispatch).
  4. Else DENY naming every arch whose most recent measurement is not
     `success`, each with its own `gh run rerun <run_id> --failed` remedy
     (re-running one failed leg is one GPU pod, not a fresh whole-workflow
     dispatch).

Any HTTP/JSON error is a hard DENY (exit 1) — never vacuous: a network
hiccup must never read as "nothing to prove" and let something silently
promote.

`fetch(url, token) -> dict` is injected so `test_gpu_prove_verdict.py` can
drive every branch above with a fake clock and a fake API, no network.

Run (real use, inside `_gpu-proof-required.yml`):
  GITHUB_TOKEN=<token> python3 ci/scripts/gpu_prove_verdict.py \\
      --repo owner/repo --sha <sha> --deadline-minutes 355
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def list_jobs(fetch: FetchFn, token: str, repo: str, run_id: int) -> list[dict]:
    """`filter=latest` -- the latest ATTEMPT of each job in the run, so a
    `gh run rerun <id> --failed` supersedes a stale attempt in place rather
    than leaving a first-attempt failure to be read as a fresh one."""
    url = f"{API_BASE}/repos/{repo}/actions/runs/{run_id}/jobs?filter=latest"
    return _paginated(fetch, token, url, "jobs")


def collect_measurements(
    fetch: FetchFn,
    token: str,
    repo: str,
    runs: list[dict],
    required_arches: list[str],
) -> tuple[dict[str, list[Measurement]], bool]:
    """Returns `({arch: [Measurement, ...]}, any_in_progress)`.

    A job is a measurement for its arch iff its name matches
    `JOB_NAME_TEMPLATE` for a required arch, its conclusion is present and
    not `"skipped"`, and it has a `completed_at` (a job with no
    `completed_at` is still running -- excluded as a measurement and folds
    into `any_in_progress`, same as a run whose own `status` is not yet
    `"completed"`)."""
    by_arch: dict[str, list[Measurement]] = {a: [] for a in required_arches}
    name_to_arch = {JOB_NAME_TEMPLATE.format(arch=a): a for a in required_arches}
    any_in_progress = any(r.get("status") != "completed" for r in runs)
    for run in runs:
        run_id = run.get("id")
        jobs = list_jobs(fetch, token, repo, run_id)
        for job in jobs:
            arch = name_to_arch.get(job.get("name"))
            if arch is None:
                continue
            conclusion = job.get("conclusion")
            if conclusion == "skipped":
                continue  # no measurement was made (rule F)
            completed_at = job.get("completed_at")
            if completed_at is None:
                any_in_progress = True
                continue
            by_arch[arch].append(
                Measurement(
                    arch=arch,
                    run_id=run_id,
                    job_id=job.get("id"),
                    conclusion=conclusion,
                    completed_at=completed_at,
                    html_url=job.get("html_url") or "",
                )
            )
    return by_arch, any_in_progress


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
) -> tuple[Verdict, bool]:
    runs = list_runs(fetch, token, repo, workflow, sha)
    by_arch, any_in_progress = collect_measurements(fetch, token, repo, runs, required_arches)
    return evaluate(by_arch, required_arches), any_in_progress


def required_arches() -> list[str]:
    return sorted(gpu_parity_matrix.load_shipped_cuda_silicon())


def run(
    *,
    repo: str,
    sha: str,
    workflow: str,
    deadline_minutes: float,
    poll_seconds: float,
    no_wait: bool,
    fetch: FetchFn,
    token: str,
    arches: list[str],
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    deadline = now() + deadline_minutes * 60.0
    verdict: Verdict
    any_in_progress = False
    while True:
        try:
            verdict, any_in_progress = check_once(fetch, token, repo, workflow, sha, arches)
        except VerdictError as e:
            print(f"::error::gpu-prove-verdict: {e}", file=err)
            return 1
        if verdict.ok:
            for arch in sorted(verdict.proofs):
                m = verdict.proofs[arch]
                print(f"PROVEN {arch}: run={m.run_id} job={m.job_id} {m.html_url}", file=out)
            return 0
        if no_wait:
            break
        if any_in_progress and now() < deadline:
            sleep(poll_seconds)
            continue
        break

    # DENY -- operator direction (2026-09-03): the "no candidate" case never
    # waits for a grace window (nothing auto-starts a prove run); a
    # red/revoked arch re-runs its OWN failed leg, not the whole workflow.
    if verdict.missing:
        print(
            f"::error::gpu-prove-verdict: DENY -- no GPU-prove measurement for "
            f"{', '.join(verdict.missing)} at {sha} -- dispatch the prove lane: "
            f"gh workflow run {workflow} --ref <tag-or-branch-to-prove>, then re-run "
            f"this workflow's failed jobs.",
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
    ap.add_argument("--deadline-minutes", type=float, default=340.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument(
        "--no-wait",
        action="store_true",
        help="Check once and return immediately -- never polls. For hand use.",
    )
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
        deadline_minutes=args.deadline_minutes,
        poll_seconds=args.poll_seconds,
        no_wait=args.no_wait,
        fetch=default_fetch,
        token=token,
        arches=required_arches(),
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
