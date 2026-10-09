#!/usr/bin/env python3
"""Tests for `proven_artifacts.py` (which run a release promotes from).

Pure Python, no network: every test drives the real `resolve()` against
`test_verdict.py`'s fake GitHub REST surface, extended with a run's artifact
list — never a stand-in for the verdict or the artifact lookup.

Covers: a proven tree names the run whose summary is its most recent green
measurement, and checks that run's artifacts; an unmeasured tree, a red most
recent measurement, an absent artifact, an expired artifact, an empty request
and an unreadable record each DENY, naming what is wrong.

Run directly: `python3 ci/scripts/test_proven_artifacts.py`
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proven_artifacts as pa  # noqa: E402
import test_verdict as tv  # noqa: E402
import verdict as vd  # noqa: E402

WHEELS = ["wheels-native-linux-x86_64", "wheels-native-macos-arm64"]


class World(tv.World):
    """`test_verdict`'s fake, plus each run's own artifacts."""

    def __init__(self):
        super().__init__()
        self.uploads: dict[int, list[dict]] = {}

    def ci_run(self, run_id: int, conclusion: str, completed_at: str, uploads=WHEELS, expired=()):
        self.bind(
            tv.run_obj(run_id, workflow=vd.CI_WORKFLOW),
            [tv.named_job(vd.CI_SUMMARY_JOB, conclusion, completed_at, html_url=f"https://x/{run_id}")],
        )
        self.uploads[run_id] = [{"name": n, "expired": n in expired} for n in uploads]

    def cookbook_run(self, run_id: int, conclusion: str, completed_at: str, uploads=("cookbook-book",)):
        self.bind(
            tv.run_obj(run_id, workflow=vd.COOKBOOK_WORKFLOW),
            [tv.named_job(vd.COOKBOOK_JOB, conclusion, completed_at, html_url=f"https://x/{run_id}")],
        )
        self.uploads[run_id] = [{"name": n, "expired": False} for n in uploads]

    def fetch(self, url: str, token: str) -> dict:
        m = re.search(r"/actions/runs/(\d+)/artifacts", url)
        if m:
            self.fetch_calls.append(url)
            if url in self.fail_urls:
                raise RuntimeError("simulated transport failure")
            return {"artifacts": self.uploads.get(int(m.group(1)), [])}
        return super().fetch(url, token)


def _resolve(world: World, names=WHEELS, lane="ci"):
    out, err = io.StringIO(), io.StringIO()
    rc = pa.resolve(
        repo=tv.REPO, tree=tv.TREE, lane=lane, names=list(names), fetch=world.fetch, token="tok", out=out, err=err
    )
    return rc, out.getvalue(), err.getvalue()


class ResolveTest(unittest.TestCase):
    def test_a_proven_tree_names_its_run(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        rc, out, _ = _resolve(w)
        self.assertEqual((rc, out.split()), (0, ["run=7"]))

    def test_the_most_recent_measurement_is_the_source(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        w.ci_run(9, "success", "2026-01-02T00:00:00Z")
        rc, out, _ = _resolve(w)
        self.assertEqual((rc, out.split()), (0, ["run=9"]))
        self.assertTrue(any(c.endswith("/actions/runs/9/artifacts") or "/actions/runs/9/artifacts?" in c for c in w.fetch_calls))

    def test_an_unmeasured_tree_denies(self):
        rc, out, err = _resolve(World())
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("has no ci.yml run", err)

    def test_a_red_most_recent_measurement_denies(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        w.ci_run(9, "failure", "2026-01-02T00:00:00Z")
        rc, out, err = _resolve(w)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("'failure' (run=9", err)

    def test_an_absent_artifact_denies_by_name(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z", uploads=WHEELS[:1])
        rc, out, err = _resolve(w)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn(f"no artifact {WHEELS[1]!r}", err)

    def test_an_expired_artifact_denies_with_the_remedy(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z", expired=WHEELS[:1])
        rc, out, err = _resolve(w)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("has expired", err)
        self.assertIn("measure the tree again", err)

    def test_an_empty_request_denies(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        rc, _, err = _resolve(w, names=[])
        self.assertEqual(rc, 1)
        self.assertIn("no artifact named", err)

    def test_an_unreadable_record_denies(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        w.fail_urls.add(f"{pa.API_BASE}/repos/{tv.REPO}/actions/runs/7/artifacts?per_page={pa.github_api.PER_PAGE}&page=1")
        rc, out, err = _resolve(w)
        self.assertEqual((rc, out), (1, ""))
        self.assertIn("::error::", err)


class LaneTest(unittest.TestCase):
    def test_every_lane_is_measured_by_one_job(self):
        for lane, requirement in pa.LANES.items():
            self.assertEqual(len(requirement().jobs), 1, lane)

    def test_the_cookbook_lane_names_its_own_run(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z")
        w.cookbook_run(8, "success", "2026-01-01T01:00:00Z")
        rc, out, _ = _resolve(w, names=["cookbook-book"], lane="cookbook-gpu")
        self.assertEqual((rc, out.split()), (0, ["run=8"]))

    def test_a_ci_run_never_proves_the_cookbook_lane(self):
        w = World()
        w.ci_run(7, "success", "2026-01-01T00:00:00Z", uploads=["cookbook-book"])
        rc, out, err = _resolve(w, names=["cookbook-book"], lane="cookbook-gpu")
        self.assertEqual((rc, out), (1, ""))
        self.assertIn(f"has no {vd.COOKBOOK_WORKFLOW} run", err)


if __name__ == "__main__":
    unittest.main()
