#!/usr/bin/env python3
"""Tests for `gpu_prove_verdict.py` (esc-084, issue #454; check-once/
fail-loud, operator direction 2026-09-03).

Pure Python, no network: every test drives the real `run()`/`evaluate()`/
`list_runs()`/`collect_measurements()` entry points against an injected fake
`fetch` — never a hand-rolled stand-in for the verdict logic itself.

Covers esc-084 control (b): every degenerate record DENIES, individually;
the positive record set PASSES and prints run/job ids per arch; the
recency/revocation rule (a later red measurement revokes an earlier green,
by `completed_at` with a run-id tiebreak); check-once semantics (an
in-progress run at this sha is invisible — it never delays or satisfies the
check; a missing measurement DENIES immediately with the dispatch remedy,
no grace window, no poll); pagination; and that any HTTP/JSON error DENIES
(never vacuously passes).

Run directly: `python3 ci/scripts/test_gpu_prove_verdict.py`
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_prove_verdict as gpv  # noqa: E402

REPO = "f-inverse/jammi-ai"
SHA = "a" * 40
WORKFLOW = gpv.DEFAULT_WORKFLOW
ARCHES = ["sm_80", "sm_86"]


def job(arch: str, conclusion, completed_at, job_id: int = 1, html_url: str = "https://x/y"):
    return {
        "name": gpv.JOB_NAME_TEMPLATE.format(arch=arch),
        "conclusion": conclusion,
        "completed_at": completed_at,
        "id": job_id,
        "html_url": html_url,
    }


# Advisory A5 fix: `list_runs` now requires `path` to be PRESENT and equal
# the exact `.github/workflows/<workflow>` path, so `run_obj`'s default
# stamps a REAL, correct path -- exactly what every production API record
# carries -- so every pre-existing fixture keeps passing that filter
# without having to spell it out. `path=None` explicitly (rather than
# omitted) reproduces the "no `path` key at all" degenerate case A5 adds a
# test for; any other string reproduces a wrong/nested path.
_DEFAULT_PATH = object()


def run_obj(run_id: int, status: str = "completed", sha: str = SHA, path=_DEFAULT_PATH):
    o = {"id": run_id, "status": status, "head_sha": sha}
    if path is _DEFAULT_PATH:
        o["path"] = f".github/workflows/{WORKFLOW}"
    elif path is not None:
        o["path"] = path
    return o


class World:
    """A tiny fake GitHub REST surface: `runs` and `jobs_by_run` are served
    paginated (honoring `gpu_prove_verdict.PER_PAGE`, which a test may
    monkeypatch smaller to exercise pagination without 100+ fixture rows)."""

    def __init__(self):
        self.runs: list[dict] = []
        self.jobs_by_run: dict[int, list[dict]] = {}
        self.fetch_calls: list[str] = []
        self.fail_urls: set[str] = set()

    def fetch(self, url: str, token: str) -> dict:
        self.fetch_calls.append(url)
        if url in self.fail_urls:
            raise RuntimeError("simulated transport failure")
        m = re.search(r"[?&]page=(\d+)", url)
        page = int(m.group(1)) if m else 1
        per_page = gpv.PER_PAGE
        if "/jobs?" in url:
            run_id = int(re.search(r"/actions/runs/(\d+)/jobs", url).group(1))
            items = self.jobs_by_run.get(run_id, [])
            key = "jobs"
        else:
            items = self.runs
            key = "workflow_runs"
        start = (page - 1) * per_page
        return {key: items[start : start + per_page]}


def _run_once(world: World, arches=ARCHES, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    rc = gpv.run(
        repo=REPO,
        sha=SHA,
        workflow=WORKFLOW,
        fetch=world.fetch,
        token="tok",
        arches=arches,
        out=out,
        err=err,
        **kwargs,
    )
    return rc, out.getvalue(), err.getvalue()


class DegenerateRecordsDenyEachOwnTest(unittest.TestCase):
    """esc-084 control (b): every degenerate record DENIES, each its own
    test. All-success control lives in PositiveCaseTest below."""

    def _good_jobs(self, run_id=1):
        return {
            run_id: [
                job("sm_80", "success", "2026-01-01T00:00:00Z", job_id=10),
                job("sm_86", "success", "2026-01-01T00:00:00Z", job_id=11),
            ]
        }

    def test_arch_absent_from_every_run(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z")]}  # sm_86 never mentioned
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_86", err)

    def test_arch_job_absent_from_the_run_though_run_exists(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_86", err)

    def test_conclusion_missing_key(self):
        w = World()
        w.runs = [run_obj(1)]
        j = job("sm_86", "success", "2026-01-01T00:00:00Z")
        del j["conclusion"]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), j]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_null(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", None, "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_non_string(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", 1, "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_empty_string(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_literal_string_none(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "None", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_skipped_is_no_measurement_not_a_pass(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "skipped", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_86", err)

    def test_conclusion_cancelled(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "cancelled", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_neutral(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "neutral", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_timed_out(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "timed_out", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_conclusion_action_required(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "action_required", "2026-01-01T00:00:00Z")]}
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_run_head_sha_differs_is_dropped_client_side(self):
        w = World()
        # A run reported at our query URL but whose OWN head_sha field
        # disagrees -- must never be trusted even though the query param
        # asked the server to scope by head_sha (esc-084 control b).
        w.runs = [run_obj(1, sha="b" * 40)]
        w.jobs_by_run = self._good_jobs(1)
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1, "a foreign-sha run must never supply a measurement")

    def test_run_of_a_different_workflow_path_is_dropped(self):
        w = World()
        w.runs = [run_obj(1, path=".github/workflows/some-other-workflow.yml")]
        w.jobs_by_run = self._good_jobs(1)
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)

    def test_run_with_no_path_key_at_all_is_dropped(self):
        # Advisory A5 fix: `"path" not in r` used to be a VACUOUS accept --
        # a run record missing the `path` key entirely must be REFUSED, not
        # trusted just because it also carries the right head_sha.
        w = World()
        w.runs = [run_obj(1, path=None)]
        w.jobs_by_run = self._good_jobs(1)
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1, "a run record with no path key at all must never supply a measurement")

    def test_run_with_nested_path_is_dropped(self):
        # Advisory A5 fix: `endswith` used to accept a path like
        # `vendor/.github/workflows/<workflow>` -- the match must be EXACT
        # equality against `.github/workflows/<workflow>`, not a suffix.
        w = World()
        w.runs = [run_obj(1, path=f"vendor/.github/workflows/{WORKFLOW}")]
        w.jobs_by_run = self._good_jobs(1)
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1, "a nested path must never satisfy the exact-path check via endswith")

    def test_latest_attempt_wins_over_an_earlier_attempt_in_same_run(self):
        # filter=latest semantics: our fake jobs endpoint only ever returns
        # the CALLER-supplied (i.e. already-latest) job list, so an arch
        # whose latest attempt succeeded after an earlier attempt failed is
        # represented directly as a single success record -- pin that this
        # single record is enough to PASS for that arch (no separate
        # "first attempt" record ever gets consulted).
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:05:00Z"), job("sm_86", "success", "2026-01-01T00:05:00Z")]}
        rc, _, _ = _run_once(w)
        self.assertEqual(rc, 0)

    def test_latest_attempt_fails_in_same_run_denies(self):
        # BLOCK B6 audit fix (esc-084 control b): the DENY direction of the
        # SAME filter=latest semantics the PASS-direction test above
        # covers -- "the arch's latest attempt failed after an earlier
        # attempt succeeded within the same run" MUST deny. The fake
        # `fetch` returns a FAILED job list whenever the request URL
        # carries `filter=latest` and a SUCCESS list otherwise -- proving
        # the consumer never reads the (unfiltered) all-attempts view.
        class LatestFailsFetch:
            def __call__(self, url, token):
                if "/jobs?" in url:
                    if "filter=latest" in url:
                        return {
                            "jobs": [
                                job("sm_80", "failure", "2026-01-01T00:05:00Z"),
                                job("sm_86", "failure", "2026-01-01T00:05:00Z"),
                            ]
                        }
                    return {
                        "jobs": [
                            job("sm_80", "success", "2026-01-01T00:00:00Z"),
                            job("sm_86", "success", "2026-01-01T00:00:00Z"),
                        ]
                    }
                return {"workflow_runs": [run_obj(1)]}

        out, err = io.StringIO(), io.StringIO()
        rc = gpv.run(
            repo=REPO, sha=SHA, workflow=WORKFLOW,
            fetch=LatestFailsFetch(), token="tok", arches=ARCHES,
            out=out, err=err,
        )
        self.assertEqual(rc, 1, "the consumer must read ONLY the filter=latest view, never the all-attempts one")

    def test_list_jobs_requests_filter_latest(self):
        # BLOCK B6 audit fix: a direct assertion that `list_jobs` requests
        # `?filter=latest` -- the fake records the URL it was actually
        # called with, rather than inferring the semantics from behavior.
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = self._good_jobs(1)
        gpv.list_jobs(w.fetch, "tok", REPO, 1)
        jobs_urls = [u for u in w.fetch_calls if "/jobs?" in u]
        self.assertTrue(jobs_urls, "list_jobs made no /jobs? request at all")
        self.assertTrue(
            all("filter=latest" in u for u in jobs_urls),
            f"list_jobs must request filter=latest on every page; got {jobs_urls}",
        )

    def test_zero_required_arches_denies_never_vacuous(self):
        # Advisory A6 fix: `evaluate(by_arch, [])` used to return `ok=True`
        # (`not [] and not {}` is vacuously True) -- a caller with zero
        # required arches asked a malformed question and must be denied,
        # never silently treated as "everything proven".
        with self.assertRaises(gpv.VerdictError):
            gpv.evaluate({}, [])
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = self._good_jobs(1)
        rc, _, err = _run_once(w, arches=[])
        self.assertEqual(rc, 1)

    def test_no_runs_at_all(self):
        w = World()
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)
        self.assertIn("sm_86", err)

    def test_http_error_denies_never_vacuous(self):
        w = World()
        url = f"{gpv.API_BASE}/repos/{REPO}/actions/workflows/{WORKFLOW}/runs?head_sha={SHA}&per_page={gpv.PER_PAGE}&page=1"
        w.fail_urls.add(url)
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("gpu-prove-verdict", err)

    def test_json_shape_error_denies(self):
        class BadFetch:
            def __call__(self, url, token):
                return {"unexpected": []}

        out, err = io.StringIO(), io.StringIO()
        rc = gpv.run(
            repo=REPO, sha=SHA, workflow=WORKFLOW,
            fetch=BadFetch(), token="tok", arches=ARCHES,
            out=out, err=err,
        )
        self.assertEqual(rc, 1)


class PositiveCaseTest(unittest.TestCase):
    def test_all_arches_success_passes_and_prints_ids(self):
        w = World()
        w.runs = [run_obj(1)]
        w.jobs_by_run = {
            1: [
                job("sm_80", "success", "2026-01-01T00:00:00Z", job_id=10, html_url="https://x/10"),
                job("sm_86", "success", "2026-01-01T00:00:00Z", job_id=11, html_url="https://x/11"),
            ]
        }
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 0, err)
        self.assertIn("sm_80", out)
        self.assertIn("run=1", out)
        self.assertIn("job=10", out)
        self.assertIn("sm_86", out)
        self.assertIn("job=11", out)


class RevocationAndRecencyTest(unittest.TestCase):
    """esc-084 control (b): recency by completed_at with a run-id tiebreak;
    a later red measurement revokes an earlier green."""

    def test_later_completed_run_revokes_an_earlier_green(self):
        w = World()
        w.runs = [run_obj(1), run_obj(2)]
        w.jobs_by_run = {
            1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            2: [job("sm_80", "failure", "2026-01-02T00:00:00Z"), job("sm_86", "success", "2026-01-02T00:00:00Z")],
        }
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)

    def test_fresh_red_on_an_OLD_run_id_revokes_a_stale_green_on_a_NEWER_id(self):
        # run 100 (numerically lower/older id) was rerun and its latest
        # attempt completed AFTER run 200's own completion -- recency must
        # follow completed_at, never run id.
        w = World()
        w.runs = [run_obj(100), run_obj(200)]
        w.jobs_by_run = {
            100: [job("sm_80", "failure", "2026-01-03T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            200: [job("sm_80", "success", "2026-01-02T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
        }
        rc, _, err = _run_once(w)
        self.assertEqual(rc, 1, "the fresher red measurement (run 100) must revoke the older green (run 200)")
        self.assertIn("sm_80", err)

    def test_fresh_green_on_an_OLD_run_id_after_a_red_on_a_NEWER_id_passes(self):
        w = World()
        w.runs = [run_obj(100), run_obj(200)]
        w.jobs_by_run = {
            100: [job("sm_80", "success", "2026-01-03T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            200: [job("sm_80", "failure", "2026-01-02T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
        }
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 0, err)
        self.assertIn("run=100", out)

    def test_tiebreak_on_equal_completed_at_prefers_higher_run_id(self):
        w = World()
        w.runs = [run_obj(1), run_obj(2)]
        w.jobs_by_run = {
            1: [job("sm_80", "failure", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            2: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
        }
        rc, out, _ = _run_once(w)
        self.assertEqual(rc, 0)
        self.assertIn("run=2", out)


class CheckOnceSemanticsTest(unittest.TestCase):
    """Check-once, fail-loud (operator direction 2026-09-03 — supersedes the
    earlier poll/wait design): green most-recent measurement passes
    immediately; an in-progress run is invisible to the check and never
    delays or satisfies it; no measurement at all DENIES immediately with
    the dispatch remedy; an all-completed red leg DENIES with the rerun
    remedy. There is exactly one lookup per run — never a second call to
    check whether something changed."""

    def test_green_most_recent_passes_immediately_even_with_a_newer_in_progress_run(self):
        w = World()
        w.runs = [run_obj(1, status="completed"), run_obj(2, status="in_progress")]
        w.jobs_by_run = {
            1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            2: [],  # the newer run has not produced any job yet
        }
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(w.fetch_calls), 1 + 2, "exactly one runs-list call plus one jobs call per run, no more")

    def test_in_progress_run_does_not_delay_or_satisfy_the_check(self):
        # A run still in_progress at this sha contributes no measurement for
        # any arch and must never be waited on -- the check-once contract
        # means this DENIES on the first (and only) lookup, not after some
        # poll interval.
        w = World()
        w.runs = [run_obj(1, status="in_progress")]
        w.jobs_by_run = {1: [job("sm_80", "failure", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)

    def test_no_measurement_denies_immediately_with_the_dispatch_remedy(self):
        w = World()
        # completely empty world: no runs at all.
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("dispatch the prove lane", err)
        self.assertIn("gh workflow run", err)

    def test_all_completed_with_a_red_leg_denies_with_the_rerun_remedy(self):
        w = World()
        w.runs = [run_obj(1, status="completed")]
        w.jobs_by_run = {1: [job("sm_80", "failure", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 1)
        self.assertIn("gh run rerun 1 --failed", err)


class PaginationTest(unittest.TestCase):
    def test_runs_list_paginates_across_pages(self):
        old_per_page = gpv.PER_PAGE
        gpv.PER_PAGE = 2
        try:
            w = World()
            w.runs = [run_obj(i) for i in range(1, 6)]  # 5 runs, PER_PAGE=2 -> 3 pages
            got = gpv.list_runs(w.fetch, "tok", REPO, WORKFLOW, SHA)
            self.assertEqual(sorted(r["id"] for r in got), [1, 2, 3, 4, 5])
            self.assertEqual(got[0]["id"], 5, "explicitly sorted by run id descending")
            page_urls = [c for c in w.fetch_calls if "page=" in c]
            self.assertGreaterEqual(len(page_urls), 3)
        finally:
            gpv.PER_PAGE = old_per_page

    def test_jobs_list_paginates_across_pages(self):
        old_per_page = gpv.PER_PAGE
        gpv.PER_PAGE = 1
        try:
            w = World()
            w.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}
            got = gpv.list_jobs(w.fetch, "tok", REPO, 1)
            self.assertEqual(len(got), 2)
        finally:
            gpv.PER_PAGE = old_per_page


if __name__ == "__main__":
    unittest.main()
