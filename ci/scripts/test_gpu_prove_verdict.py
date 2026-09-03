#!/usr/bin/env python3
"""Tests for `gpu_prove_verdict.py` (esc-084, issue #454).

Pure Python, no network: every test drives the real `run()`/`evaluate()`/
`list_runs()`/`collect_measurements()` entry points against an injected fake
`fetch` (and, for the poll/wait state machine, a fake clock/`sleep`) — never
a hand-rolled stand-in for the verdict logic itself.

Covers esc-084 control (b): every degenerate record DENIES, individually;
the positive record set PASSES and prints run/job ids per arch; the
recency/revocation rule (amendments F/V); the wait state machine
(amendments Q/S); pagination; and that any HTTP/JSON error DENIES (never
vacuously passes).

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


def run_obj(run_id: int, status: str = "completed", sha: str = SHA, path: str | None = None):
    o = {"id": run_id, "status": status, "head_sha": sha}
    if path is not None:
        o["path"] = path
    return o


class World:
    """A tiny fake GitHub REST surface: `runs` and `jobs_by_run` are served
    paginated (honoring `gpu_prove_verdict.PER_PAGE`, which a test may
    monkeypatch smaller to exercise pagination without 100+ fixture rows).
    `sleep` is injectable per-test so a poll tick can mutate the world
    in-place (simulating "the leg finished while we were waiting")."""

    def __init__(self):
        self.runs: list[dict] = []
        self.jobs_by_run: dict[int, list[dict]] = {}
        self.fetch_calls: list[str] = []
        self.now_val = 0.0
        self.fail_urls: set[str] = set()
        self.on_sleep = None  # optional callable(world) -> None

    def now(self) -> float:
        return self.now_val

    def sleep(self, seconds: float) -> None:
        self.now_val += seconds
        if self.on_sleep is not None:
            self.on_sleep(self)

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
        deadline_minutes=kwargs.pop("deadline_minutes", 10.0),
        poll_seconds=kwargs.pop("poll_seconds", 1.0),
        no_wait=kwargs.pop("no_wait", False),
        fetch=world.fetch,
        token="tok",
        arches=arches,
        sleep=world.sleep,
        now=world.now,
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
            repo=REPO, sha=SHA, workflow=WORKFLOW, deadline_minutes=1.0, poll_seconds=1.0,
            no_wait=False, fetch=BadFetch(), token="tok", arches=ARCHES,
            sleep=lambda s: None, now=lambda: 0.0, out=out, err=err,
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
    """Amendment F/V: recency by completed_at with a run-id tiebreak."""

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


class WaitStateMachineTest(unittest.TestCase):
    """Amendments Q/S: green most-recent wins immediately even with newer
    in-progress runs; red-most-recent-plus-in-progress polls; no candidate
    denies immediately (no grace); all-completed-with-a-red-leg fails fast
    (never polls); a deadline is honored; `--no-wait` never polls."""

    def test_green_most_recent_passes_immediately_even_with_a_newer_in_progress_run(self):
        w = World()
        w.runs = [run_obj(1, status="completed"), run_obj(2, status="in_progress")]
        w.jobs_by_run = {
            1: [job("sm_80", "success", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")],
            2: [],  # the newer run has not produced any job yet
        }
        rc, out, err = _run_once(w)
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(w.fetch_calls), 1 + 2, "no poll: one runs-list call plus one jobs call per run, no more")

    def test_red_most_recent_plus_in_progress_polls_then_resolves(self):
        w = World()
        w.runs = [run_obj(1, status="in_progress")]
        w.jobs_by_run = {1: [job("sm_80", "failure", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}

        def on_sleep(world: World) -> None:
            world.runs = [run_obj(1, status="completed")]
            world.jobs_by_run = {1: [job("sm_80", "success", "2026-01-01T00:10:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}

        w.on_sleep = on_sleep
        rc, out, err = _run_once(w, deadline_minutes=10.0, poll_seconds=1.0)
        self.assertEqual(rc, 0, err)
        self.assertIn("run=1", out)

    def test_no_candidate_denies_immediately_no_grace(self):
        w = World()
        # completely empty world: no runs at all, nothing in progress.
        rc, out, err = _run_once(w, deadline_minutes=10.0, poll_seconds=1.0)
        self.assertEqual(rc, 1)
        self.assertEqual(w.now_val, 0.0, "no sleep must ever be called for the no-candidate case (no grace, per S)")
        self.assertIn("dispatch the prove lane", err)

    def test_all_completed_with_a_red_leg_fails_fast_never_polls(self):
        w = World()
        w.runs = [run_obj(1, status="completed")]
        w.jobs_by_run = {1: [job("sm_80", "failure", "2026-01-01T00:00:00Z"), job("sm_86", "success", "2026-01-01T00:00:00Z")]}
        rc, out, err = _run_once(w, deadline_minutes=10.0, poll_seconds=1.0)
        self.assertEqual(rc, 1)
        self.assertEqual(w.now_val, 0.0, "an all-completed red leg must never poll")
        self.assertIn("gh run rerun 1 --failed", err)

    def test_deadline_reached_denies(self):
        w = World()
        w.runs = [run_obj(1, status="in_progress")]
        w.jobs_by_run = {1: []}
        rc, out, err = _run_once(w, deadline_minutes=0.01, poll_seconds=1.0)  # 0.6s deadline, 1s poll
        self.assertEqual(rc, 1)
        self.assertGreaterEqual(w.now_val, 1.0, "at least one poll tick must have elapsed before the deadline broke the loop")

    def test_no_wait_never_polls(self):
        w = World()
        w.runs = [run_obj(1, status="in_progress")]
        w.jobs_by_run = {1: [job("sm_80", "failure", "2026-01-01T00:00:00Z")]}
        rc, out, err = _run_once(w, no_wait=True, deadline_minutes=10.0, poll_seconds=1.0)
        self.assertEqual(rc, 1)
        self.assertEqual(w.now_val, 0.0, "--no-wait must never sleep")


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
