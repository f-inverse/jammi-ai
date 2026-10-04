#!/usr/bin/env python3
"""Tests for `verdict.py` (check-once, fail-loud).

Pure Python, no network: every test drives the real `require()`/`probe()`/
`evaluate()`/`bound_runs()`/`collect_measurements()` entry points against an
injected fake `fetch` — never a hand-rolled stand-in for the verdict logic.

Covers: a run binds to a tree only through its `subject-<tree>` artifact and
only as a run of the requirement's own workflow; every degenerate record
DENIES, individually; the positive record set PASSES and prints run/job ids;
the recency/revocation rule (a later red measurement revokes an earlier
green, by `completed_at` with a run-id tiebreak); check-once semantics (a run
still in progress is invisible — it never delays or satisfies the check);
pagination; that any HTTP/JSON error DENIES the release gate; and that the
plan's probe reads an unreadable record as "measure again", never as proven.

Run directly: `python3 ci/scripts/test_verdict.py`
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import github_api  # noqa: E402
import verdict as vd  # noqa: E402

REPO = "f-inverse/jammi-ai"
TREE = "a" * 40
ARCHES = ["sm_80", "sm_86"]
GPU = vd.Requirement(
    vd.GPU_WORKFLOW,
    tuple(vd.GPU_JOB_TEMPLATE.format(arch=a) for a in ARCHES),
    vd.gpu_requirement().remedy,
)
CI = vd.ci_requirement()
COOKBOOK = vd.cookbook_requirement()


def gpu_job(arch: str, conclusion, completed_at, job_id: int = 1, html_url: str = "https://x/y"):
    return named_job(vd.GPU_JOB_TEMPLATE.format(arch=arch), conclusion, completed_at, job_id, html_url)


def named_job(name: str, conclusion, completed_at, job_id: int = 1, html_url: str = "https://x/y"):
    return {"name": name, "conclusion": conclusion, "completed_at": completed_at, "id": job_id, "html_url": html_url}


_DEFAULT_PATH = object()


def run_obj(run_id: int, workflow: str = vd.GPU_WORKFLOW, status: str = "completed", path=_DEFAULT_PATH):
    """A run record. `path=None` omits the key entirely; any other string is a
    wrong or nested path."""
    o = {"id": run_id, "status": status}
    if path is _DEFAULT_PATH:
        o["path"] = f".github/workflows/{workflow}"
    elif path is not None:
        o["path"] = path
    return o


def subject(run_id: int, tree: str = TREE, name: str | None = None):
    return {"name": name or f"subject-{tree}", "workflow_run": {"id": run_id}}


class World:
    """A fake GitHub REST surface: the artifacts named by tree, each run's own
    record, and each run's jobs, all served paginated under
    `github_api.PER_PAGE` (a test may shrink it to exercise pagination)."""

    def __init__(self):
        self.artifacts: list[dict] = []
        self.runs: dict[int, dict] = {}
        self.jobs_by_run: dict[int, list[dict]] = {}
        self.fetch_calls: list[str] = []
        self.fail_urls: set[str] = set()

    def bind(self, run: dict, jobs: list[dict], tree: str = TREE):
        self.runs[run["id"]] = run
        self.jobs_by_run[run["id"]] = jobs
        self.artifacts.append(subject(run["id"], tree))

    def fetch(self, url: str, token: str) -> dict:
        self.fetch_calls.append(url)
        if url in self.fail_urls:
            raise RuntimeError("simulated transport failure")
        m = re.search(r"[?&]page=(\d+)", url)
        page = int(m.group(1)) if m else 1
        per_page = github_api.PER_PAGE
        start = (page - 1) * per_page
        if "/jobs?" in url:
            run_id = int(re.search(r"/actions/runs/(\d+)/jobs", url).group(1))
            return {"jobs": self.jobs_by_run.get(run_id, [])[start : start + per_page]}
        if "/actions/artifacts?" in url:
            name = re.search(r"name=([^&]+)", url).group(1)
            matching = [a for a in self.artifacts if a.get("name") == name]
            return {"artifacts": matching[start : start + per_page]}
        m = re.search(r"/actions/runs/(\d+)$", url)
        if m:
            return self.runs[int(m.group(1))]
        raise AssertionError(f"unexpected url {url}")


def _require(world, requirements=(GPU,), tree=TREE):
    out, err = io.StringIO(), io.StringIO()
    rc = vd.require(
        repo=REPO, tree=tree, requirements=list(requirements), fetch=world.fetch, token="tok", out=out, err=err
    )
    return rc, out.getvalue(), err.getvalue()


def _good(run_id: int = 1):
    return run_obj(run_id), [
        gpu_job("sm_80", "success", "2026-01-01T00:00:00Z", job_id=10),
        gpu_job("sm_86", "success", "2026-01-01T00:00:00Z", job_id=11),
    ]


def _one_leg(sm86_conclusion, completed_at="2026-01-01T00:00:00Z"):
    w = World()
    w.bind(run_obj(1), [gpu_job("sm_80", "success", "2026-01-01T00:00:00Z"), gpu_job("sm_86", sm86_conclusion, completed_at)])
    return w


class SubjectBindingTest(unittest.TestCase):
    """A run proves a tree only through the `subject-<tree>` artifact it
    recorded, and only as a run of the requirement's own workflow."""

    def test_a_run_bound_to_another_tree_never_proves_this_one(self):
        w = World()
        run, jobs = _good()
        w.bind(run, jobs, tree="b" * 40)
        rc, _, err = _require(w)
        self.assertEqual(rc, 1)
        self.assertIn("no measurement", err)

    def test_an_artifact_named_otherwise_is_dropped_client_side(self):
        # The server is asked for `name=subject-<tree>`; a record of another
        # name in the answer never binds a run.
        w = World()
        run, jobs = _good()
        w.runs[1], w.jobs_by_run[1] = run, jobs
        w.artifacts.append(subject(1, name=f"subject-{TREE}-extra"))

        def fetch(url, token):
            if "/actions/artifacts?" in url:
                return {"artifacts": w.artifacts}
            return w.fetch(url, token)

        out, err = io.StringIO(), io.StringIO()
        rc = vd.require(repo=REPO, tree=TREE, requirements=[GPU], fetch=fetch, token="tok", out=out, err=err)
        self.assertEqual(rc, 1)

    def test_an_artifact_with_no_run_id_binds_nothing(self):
        w = World()
        w.artifacts.append({"name": f"subject-{TREE}", "workflow_run": {}})
        rc, _, _ = _require(w)
        self.assertEqual(rc, 1)

    def test_a_run_of_another_workflow_never_proves_this_requirement(self):
        w = World()
        _, jobs = _good()
        w.bind(run_obj(1, workflow=vd.CI_WORKFLOW), jobs)
        rc, _, _ = _require(w)
        self.assertEqual(rc, 1)

    def test_a_run_with_no_path_key_at_all_is_dropped(self):
        w = World()
        _, jobs = _good()
        w.bind(run_obj(1, path=None), jobs)
        rc, _, _ = _require(w)
        self.assertEqual(rc, 1)

    def test_a_run_with_a_nested_path_is_dropped(self):
        w = World()
        _, jobs = _good()
        w.bind(run_obj(1, path=f"vendor/.github/workflows/{vd.GPU_WORKFLOW}"), jobs)
        rc, _, _ = _require(w)
        self.assertEqual(rc, 1)

    def test_two_artifacts_of_one_run_read_the_run_once(self):
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        w.artifacts.append(subject(1))
        rc, out, err = _require(w)
        self.assertEqual(rc, 0, err)
        self.assertEqual(sum(1 for u in w.fetch_calls if u.endswith("/actions/runs/1")), 1)


class DegenerateRecordsDenyEachOwnTest(unittest.TestCase):
    """Every degenerate record DENIES, each its own test."""

    def test_job_absent_from_every_run(self):
        w = World()
        w.bind(run_obj(1), [gpu_job("sm_80", "success", "2026-01-01T00:00:00Z")])
        rc, _, err = _require(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_86", err)

    def test_conclusion_missing_key(self):
        w = World()
        j = gpu_job("sm_86", "success", "2026-01-01T00:00:00Z")
        del j["conclusion"]
        w.bind(run_obj(1), [gpu_job("sm_80", "success", "2026-01-01T00:00:00Z"), j])
        self.assertEqual(_require(w)[0], 1)

    def test_conclusion_null(self):
        self.assertEqual(_require(_one_leg(None))[0], 1)

    def test_conclusion_non_string(self):
        self.assertEqual(_require(_one_leg(1))[0], 1)

    def test_conclusion_empty_string(self):
        self.assertEqual(_require(_one_leg(""))[0], 1)

    def test_conclusion_literal_string_none(self):
        self.assertEqual(_require(_one_leg("None"))[0], 1)

    def test_conclusion_skipped_is_no_measurement_not_a_pass(self):
        rc, _, err = _require(_one_leg("skipped"))
        self.assertEqual(rc, 1)
        self.assertIn("sm_86", err)

    def test_conclusion_cancelled(self):
        self.assertEqual(_require(_one_leg("cancelled"))[0], 1)

    def test_conclusion_neutral(self):
        self.assertEqual(_require(_one_leg("neutral"))[0], 1)

    def test_conclusion_timed_out(self):
        self.assertEqual(_require(_one_leg("timed_out"))[0], 1)

    def test_conclusion_action_required(self):
        self.assertEqual(_require(_one_leg("action_required"))[0], 1)

    def test_latest_attempt_wins_over_an_earlier_attempt_in_same_run(self):
        w = World()
        w.bind(run_obj(1), [gpu_job("sm_80", "success", "2026-01-01T00:05:00Z"), gpu_job("sm_86", "success", "2026-01-01T00:05:00Z")])
        self.assertEqual(_require(w)[0], 0)

    def test_latest_attempt_fails_in_same_run_denies(self):
        # The fake answers a FAILED job list for `filter=latest` and a SUCCESS
        # list otherwise: the reader must read only the latest view.
        w = World()
        w.bind(run_obj(1), [])

        def fetch(url, token):
            if "/jobs?" in url:
                c = "failure" if "filter=latest" in url else "success"
                return {"jobs": [gpu_job("sm_80", c, "2026-01-01T00:05:00Z"), gpu_job("sm_86", c, "2026-01-01T00:05:00Z")]}
            return w.fetch(url, token)

        out, err = io.StringIO(), io.StringIO()
        rc = vd.require(repo=REPO, tree=TREE, requirements=[GPU], fetch=fetch, token="tok", out=out, err=err)
        self.assertEqual(rc, 1)

    def test_list_jobs_requests_filter_latest(self):
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        github_api.list_jobs(w.fetch, "tok", REPO, 1)
        jobs_urls = [u for u in w.fetch_calls if "/jobs?" in u]
        self.assertTrue(jobs_urls)
        self.assertTrue(all("filter=latest" in u for u in jobs_urls), jobs_urls)

    def test_a_requirement_naming_no_job_denies_never_vacuous(self):
        with self.assertRaises(github_api.ApiError):
            vd.evaluate(vd.Requirement(vd.GPU_WORKFLOW, (), ""), {})
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        rc, _, _ = _require(w, requirements=[vd.Requirement(vd.GPU_WORKFLOW, (), "")])
        self.assertEqual(rc, 1)

    def test_no_runs_at_all(self):
        rc, _, err = _require(World())
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)
        self.assertIn("sm_86", err)

    def test_http_error_denies_never_vacuous(self):
        w = World()
        w.fail_urls.add(
            f"{github_api.API_BASE}/repos/{REPO}/actions/artifacts?name=subject-{TREE}&per_page={github_api.PER_PAGE}&page=1"
        )
        rc, _, err = _require(w)
        self.assertEqual(rc, 1)
        self.assertIn("verdict", err)

    def test_json_shape_error_denies(self):
        out, err = io.StringIO(), io.StringIO()
        rc = vd.require(
            repo=REPO, tree=TREE, requirements=[GPU], fetch=lambda u, t: {"unexpected": []}, token="tok", out=out, err=err
        )
        self.assertEqual(rc, 1)


class RunningLatestAttemptFallsBackToRunCompletedAttemptTest(unittest.TestCase):
    """`filter=latest` hides an earlier, already-COMPLETED attempt of the same
    run whenever the job's latest attempt is still in progress: a red
    completed attempt behind an in-flight rerun still denies; a green one
    still passes."""

    def _world(self, run2_all_sm80: list[dict]):
        w = World()
        w.bind(run_obj(1), [gpu_job("sm_80", "success", "2026-01-01T00:00:00Z", 1), gpu_job("sm_86", "success", "2026-01-01T00:00:00Z", 2)])
        w.bind(run_obj(2, status="in_progress"), [])
        running = gpu_job("sm_80", None, None, job_id=21)
        stable = gpu_job("sm_86", "success", "2026-01-02T00:00:00Z", job_id=22)

        def fetch(url, token):
            if "/actions/runs/2/jobs" in url:
                w.fetch_calls.append(url)
                if "filter=latest" in url:
                    return {"jobs": [running, stable]}
                return {"jobs": [*run2_all_sm80, running, stable]}
            return w.fetch(url, token)

        return w, fetch

    def _rc(self, fetch):
        out, err = io.StringIO(), io.StringIO()
        rc = vd.require(repo=REPO, tree=TREE, requirements=[GPU], fetch=fetch, token="tok", out=out, err=err)
        return rc, out.getvalue(), err.getvalue()

    def test_red_completed_attempt_behind_an_in_progress_rerun_denies(self):
        w, fetch = self._world([gpu_job("sm_80", "failure", "2026-01-02T00:00:00Z", job_id=20)])
        rc, _, err = self._rc(fetch)
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)
        self.assertTrue(any("/actions/runs/2/jobs" in u and "filter=all" in u for u in w.fetch_calls))

    def test_green_completed_attempt_behind_an_in_progress_rerun_passes(self):
        _, fetch = self._world([gpu_job("sm_80", "success", "2026-01-02T00:00:00Z", job_id=20)])
        rc, out, err = self._rc(fetch)
        self.assertEqual(rc, 0, err)
        self.assertIn("run=2", out)

    def test_no_completed_attempt_at_all_falls_back_to_the_older_run(self):
        _, fetch = self._world([])
        rc, out, err = self._rc(fetch)
        self.assertEqual(rc, 0, err)
        self.assertIn("run=1", out)


class PositiveCaseTest(unittest.TestCase):
    def test_every_job_success_passes_and_prints_ids(self):
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        rc, out, err = _require(w)
        self.assertEqual(rc, 0, err)
        for needle in ("sm_80", "sm_86", "run=1", "job=10", "job=11"):
            self.assertIn(needle, out)


class RevocationAndRecencyTest(unittest.TestCase):
    """Recency by `completed_at` with a run-id tiebreak; a later red
    measurement revokes an earlier green."""

    def _two(self, a: tuple, b: tuple, ids=(1, 2)):
        w = World()
        w.bind(run_obj(ids[0]), [gpu_job("sm_80", *a), gpu_job("sm_86", "success", "2026-01-01T00:00:00Z")])
        w.bind(run_obj(ids[1]), [gpu_job("sm_80", *b), gpu_job("sm_86", "success", "2026-01-01T00:00:00Z")])
        return w

    def test_later_completed_run_revokes_an_earlier_green(self):
        rc, _, err = _require(self._two(("success", "2026-01-01T00:00:00Z"), ("failure", "2026-01-02T00:00:00Z")))
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)

    def test_fresh_red_on_an_OLD_run_id_revokes_a_stale_green_on_a_NEWER_id(self):
        w = self._two(("failure", "2026-01-03T00:00:00Z"), ("success", "2026-01-02T00:00:00Z"), ids=(100, 200))
        self.assertEqual(_require(w)[0], 1)

    def test_fresh_green_on_an_OLD_run_id_after_a_red_on_a_NEWER_id_passes(self):
        w = self._two(("success", "2026-01-03T00:00:00Z"), ("failure", "2026-01-02T00:00:00Z"), ids=(100, 200))
        rc, out, err = _require(w)
        self.assertEqual(rc, 0, err)
        self.assertIn("run=100", out)

    def test_tiebreak_on_equal_completed_at_prefers_higher_run_id(self):
        rc, out, _ = _require(self._two(("failure", "2026-01-01T00:00:00Z"), ("success", "2026-01-01T00:00:00Z")))
        self.assertEqual(rc, 0)
        self.assertIn("run=2", out)


class CheckOnceSemanticsTest(unittest.TestCase):
    """A green most-recent measurement passes on the one lookup; a run still
    in progress is invisible and never delays or satisfies the check; no
    measurement DENIES with the dispatch remedy; a red leg DENIES with the
    rerun remedy."""

    def test_green_most_recent_passes_immediately_even_with_a_newer_in_progress_run(self):
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        w.bind(run_obj(2, status="in_progress"), [])
        rc, _, err = _require(w)
        self.assertEqual(rc, 0, err)
        # One artifacts listing, then one run record and one jobs listing per run.
        self.assertEqual(len(w.fetch_calls), 1 + 2 * 2)

    def test_in_progress_run_does_not_delay_or_satisfy_the_check(self):
        w = World()
        w.bind(run_obj(1, status="in_progress"), [gpu_job("sm_80", "failure", "2026-01-01T00:00:00Z"), gpu_job("sm_86", "success", "2026-01-01T00:00:00Z")])
        rc, _, err = _require(w)
        self.assertEqual(rc, 1)
        self.assertIn("sm_80", err)

    def test_no_gpu_measurement_denies_with_the_dispatch_remedy(self):
        rc, _, err = _require(World())
        self.assertEqual(rc, 1)
        self.assertIn("gh workflow run gpu-prove.yml", err)

    def test_no_ci_measurement_denies_naming_the_workflow_that_measures(self):
        rc, _, err = _require(World(), requirements=[CI])
        self.assertEqual(rc, 1)
        self.assertIn("ci.yml", err)

    def test_a_red_leg_denies_with_the_rerun_remedy(self):
        w = World()
        w.bind(run_obj(1), [gpu_job("sm_80", "failure", "2026-01-01T00:00:00Z"), gpu_job("sm_86", "success", "2026-01-01T00:00:00Z")])
        rc, _, err = _require(w)
        self.assertEqual(rc, 1)
        self.assertIn("gh run rerun 1 --failed", err)


class ReleaseGateTest(unittest.TestCase):
    """The release gate holds only when the tree's GPU prove, its CI and its
    cookbook all hold."""

    def _both(self, ci_conclusion):
        w = World()
        run, jobs = _good(1)
        w.bind(run, jobs)
        w.bind(run_obj(2, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, ci_conclusion, "2026-01-01T00:00:00Z", 30)])
        return w

    def _all(self, cookbook_conclusion):
        w = self._both("success")
        w.bind(
            run_obj(3, workflow=vd.COOKBOOK_WORKFLOW),
            [named_job(vd.COOKBOOK_JOB, cookbook_conclusion, "2026-01-01T00:00:00Z", 40)],
        )
        return w

    def test_a_release_requires_the_gpu_prove_ci_and_the_cookbook(self):
        self.assertEqual(
            [r.workflow for r in vd.release_requirements()],
            [vd.GPU_WORKFLOW, vd.CI_WORKFLOW, vd.COOKBOOK_WORKFLOW],
        )

    def test_every_lane_proven_passes(self):
        rc, out, err = _require(self._all("success"), requirements=[GPU, CI, COOKBOOK])
        self.assertEqual(rc, 0, err)
        self.assertIn(vd.COOKBOOK_JOB, out)

    def test_gpu_and_ci_proven_but_no_cookbook_run_denies(self):
        rc, _, err = _require(self._both("success"), requirements=[GPU, CI, COOKBOOK])
        self.assertEqual(rc, 1)
        self.assertIn(vd.COOKBOOK_WORKFLOW, err)

    def test_a_red_cookbook_run_denies(self):
        rc, _, err = _require(self._all("failure"), requirements=[GPU, CI, COOKBOOK])
        self.assertEqual(rc, 1)
        self.assertIn(vd.COOKBOOK_JOB, err)

    def test_gpu_and_ci_proven_passes(self):
        rc, out, err = _require(self._both("success"), requirements=[GPU, CI])
        self.assertEqual(rc, 0, err)
        self.assertIn(vd.CI_SUMMARY_JOB, out)

    def test_gpu_proven_but_ci_red_denies(self):
        rc, _, err = _require(self._both("failure"), requirements=[GPU, CI])
        self.assertEqual(rc, 1)
        self.assertIn(vd.CI_SUMMARY_JOB, err)

    def test_ci_proven_but_no_gpu_prove_denies(self):
        w = World()
        w.bind(run_obj(2, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "success", "2026-01-01T00:00:00Z")])
        rc, _, err = _require(w, requirements=[GPU, CI])
        self.assertEqual(rc, 1)
        self.assertIn("gpu-prove.yml", err)

    def test_a_skipped_summary_is_no_measurement(self):
        rc, _, _ = _require(self._both("skipped"), requirements=[CI])
        self.assertEqual(rc, 1)


class ProbeTest(unittest.TestCase):
    """`ci.yml`'s plan asks whether its tree is already proven. It never fails
    the run: an unreadable record means the run measures again."""

    def _probe(self, fetch):
        out, err = io.StringIO(), io.StringIO()
        rc = vd.probe(repo=REPO, tree=TREE, fetch=fetch, token="tok", out=out, err=err)
        return rc, out.getvalue(), err.getvalue()

    def test_a_proven_tree_reads_proven(self):
        w = World()
        w.bind(run_obj(7, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "success", "2026-01-01T00:00:00Z")])
        rc, out, err = self._probe(w.fetch)
        self.assertEqual((rc, out.split()), (0, ["proven=true"]))
        self.assertIn("run 7", err)

    def test_the_proving_run_is_the_most_recent_measurement(self):
        w = World()
        w.bind(run_obj(7, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "success", "2026-01-01T00:00:00Z")])
        w.bind(run_obj(9, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "success", "2026-01-02T00:00:00Z")])
        _, out, err = self._probe(w.fetch)
        self.assertEqual(out.split(), ["proven=true"])
        self.assertIn("run 9", err)

    def test_an_unmeasured_tree_reads_unproven(self):
        rc, out, _ = self._probe(World().fetch)
        self.assertEqual((rc, out.strip()), (0, "proven=false"))

    def test_a_red_summary_reads_unproven(self):
        w = World()
        w.bind(run_obj(7, workflow=vd.CI_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "failure", "2026-01-01T00:00:00Z")])
        self.assertEqual(self._probe(w.fetch)[1].strip(), "proven=false")

    def test_an_unreadable_record_reads_unproven_with_a_warning(self):
        def fetch(url, token):
            raise RuntimeError("simulated transport failure")

        rc, out, err = self._probe(fetch)
        self.assertEqual((rc, out.strip()), (0, "proven=false"))
        self.assertIn("::warning::", err)

    def test_a_gpu_prove_run_never_proves_the_ci_tree(self):
        w = World()
        w.bind(run_obj(7, workflow=vd.GPU_WORKFLOW), [named_job(vd.CI_SUMMARY_JOB, "success", "2026-01-01T00:00:00Z")])
        self.assertEqual(self._probe(w.fetch)[1].strip(), "proven=false")


class PaginationTest(unittest.TestCase):
    def setUp(self):
        self._per_page = github_api.PER_PAGE

    def tearDown(self):
        github_api.PER_PAGE = self._per_page

    def test_artifacts_paginate_and_runs_sort_newest_first(self):
        github_api.PER_PAGE = 2
        w = World()
        for i in range(1, 6):
            run, jobs = _good(i)
            w.bind(run, jobs)
        got = vd.bound_runs(w.fetch, "tok", REPO, vd.GPU_WORKFLOW, TREE)
        self.assertEqual([r["id"] for r in got], [5, 4, 3, 2, 1])
        self.assertGreaterEqual(len([c for c in w.fetch_calls if "/actions/artifacts?" in c]), 3)

    def test_jobs_paginate(self):
        github_api.PER_PAGE = 1
        w = World()
        run, jobs = _good()
        w.bind(run, jobs)
        self.assertEqual(len(github_api.list_jobs(w.fetch, "tok", REPO, 1)), 2)


if __name__ == "__main__":
    unittest.main()
