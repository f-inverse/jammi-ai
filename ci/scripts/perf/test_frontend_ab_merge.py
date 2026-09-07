#!/usr/bin/env python3
"""Fixture-directory tests for `frontend_ab_merge.py` -- the merge/bar-
decision stage `frontend_ab.sh` invokes as
`python3 "$DIR/frontend_ab_merge.py" ...`.

The real A/B rehearsal for issue #421's follow-on ("media front-end
parallelization") on pod p421b found that driver's own report-reading
defect: every one of eight `finetune-run` legs completed, but the OLD
inline-heredoc reader read `steps_measured` / `media_front_end_wall_s` /
`train_run_wall_s` off the report's TOP LEVEL, when the real report nests
them under `tiers.finetune_run` -- so every leg read back
`FAIL "report missing 'steps_measured'"` although nothing had actually
failed. `test_frontend_ab_dry_run.py`'s own hermetic suite could not catch
this because its DRY_RUN stub fabricated the SAME (wrong) flat shape the
buggy reader expected -- esc-088's class, a stub that does not mirror the
real envelope hides the exact bug a real run then hits.

`RealEnvelopeFixtureTests` below drives a REAL, committed,
envelope-trimmed cut of that same rehearsal's `htsat__tip__r1.json`
(`fixtures/frontend_ab_rehearsal/`, provenance in that directory's own
PROVENANCE.md) through `frontend_ab_merge.load_leg` -- the actual reader
`frontend_ab.sh` calls via `frontend_ab_merge.main` -- never a hand-rolled
dict standing in for what a real report looks like.
`FlatShapeRegressionTests` is the non-vacuous negative control for the bug
itself: the driver's OLD top-level assumption, replayed against the FIXED
reader, must fail loudly (never silently read as OK).

Run directly: `python3 ci/scripts/perf/test_frontend_ab_merge.py`
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frontend_ab_merge  # noqa: E402

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(PERF_DIR, "fixtures", "frontend_ab_rehearsal")
REAL_ENVELOPE_FIXTURE = os.path.join(FIXTURES_DIR, "htsat_tip_r1_envelope.json")


def _write_leg(raw_dir, tower, role, repeat, report, exit_code=0):
    p = Path(raw_dir)
    (p / f"{tower}__{role}__{repeat}.exit").write_text(str(exit_code))
    if report is not None:
        (p / f"{tower}__{role}__{repeat}.json").write_text(json.dumps(report))


class RealEnvelopeFixtureTests(unittest.TestCase):
    """Drives a REAL, committed, envelope-trimmed `finetune-run` report
    (from the pod-p421b A/B rehearsal that found this reader's own defect)
    through the real reader `frontend_ab.sh` calls -- never a hand-rolled
    dict standing in for what a real report looks like."""

    def test_real_report_cut_reads_ok_with_the_exact_source_numbers(self):
        with open(REAL_ENVELOPE_FIXTURE, encoding="utf-8") as f:
            report = json.load(f)
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(raw_dir, "htsat", "tip", "r1", report)
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "OK", leg)
        self.assertEqual(leg["steps_measured"], 100)
        self.assertEqual(leg["rayon_pool_threads"], 26)
        # Byte-identical source literals (PROVENANCE.md) -- an exact
        # equality, not a tolerance, since nothing here was rounded.
        self.assertEqual(leg["front_per_step"], 12.049923687 / 100)
        self.assertEqual(leg["train_per_step"], 35.219248152 / 100)


class FlatShapeRegressionTests(unittest.TestCase):
    """The exact bug the real A/B rehearsal on pod p421b found: a
    COMPLETE, successful `finetune-run` report whose fields sit at the
    report's top level (this driver's OLD, pre-fix shape assumption) must
    FAIL loudly, never silently read as OK with a wrong/absent tier."""

    def test_a_flat_top_level_report_is_a_named_fail_not_ok(self):
        flat_report = {
            "steps_measured": 100,
            "media_front_end_wall_s": 12.05,
            "train_run_wall_s": 35.22,
            "rayon_pool_threads": 26,
        }
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(raw_dir, "htsat", "tip", "r1", flat_report)
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "FAIL", leg)
        self.assertIn("tiers.finetune_run", leg["reason"])

    def test_a_report_with_no_tiers_object_at_all_is_a_named_fail(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(raw_dir, "htsat", "tip", "r1", {"engine_version": "0.49.1"})
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "FAIL", leg)
        self.assertIn("tiers.finetune_run", leg["reason"])


class NonFiniteControlTests(unittest.TestCase):
    """Non-vacuous negative control (family F): a NaN measurement must be
    REFUSED, never silently compared -- `NaN > 0`/`NaN <= 0` are both
    `False`, so a naive threshold check would pass a diverged leg straight
    through to the bar's own ratio arithmetic."""

    def _report_with(self, **overrides):
        tier = {
            "steps_measured": 100,
            "media_front_end_wall_s": 12.05,
            "train_run_wall_s": 35.22,
            "rayon_pool_threads": 26,
        }
        tier.update(overrides)
        return {"tiers": {"finetune_run": tier}}

    def test_a_nan_media_front_end_wall_s_is_refused_not_silently_ok(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(
                raw_dir, "htsat", "tip", "r1",
                self._report_with(media_front_end_wall_s=math.nan),
            )
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "FAIL", leg)
        self.assertIn("not finite", leg["reason"])

    def test_an_infinite_train_run_wall_s_is_refused_not_silently_ok(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(
                raw_dir, "htsat", "tip", "r1",
                self._report_with(train_run_wall_s=math.inf),
            )
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "FAIL", leg)
        self.assertIn("not finite", leg["reason"])

    def test_a_zero_steps_measured_is_refused_not_a_divide_by_zero(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            _write_leg(raw_dir, "htsat", "tip", "r1", self._report_with(steps_measured=0))
            leg = frontend_ab_merge.load_leg(Path(raw_dir), "htsat", "tip", "r1")
        self.assertEqual(leg["outcome"], "FAIL", leg)
        self.assertIn("steps_measured", leg["reason"])


class FullBuildReportTests(unittest.TestCase):
    """Drives `build_report` (the exact function `main` calls) across all
    eight legs -- the mean/ratio/bar arithmetic, not merely the single-leg
    reader."""

    def _write_all_ok(self, raw_dir, base_front=0.020, tip_front=0.0020, p=13, steps=100):
        for tower in frontend_ab_merge.TOWERS:
            for role, front in (("base", base_front), ("tip", tip_front)):
                for repeat in frontend_ab_merge.REPEATS:
                    tier = {
                        "steps_measured": steps,
                        "media_front_end_wall_s": front * steps,
                        "train_run_wall_s": 0.15 * steps,
                    }
                    if role == "tip":
                        tier["rayon_pool_threads"] = p
                    _write_leg(raw_dir, tower, role, repeat, {"tiers": {"finetune_run": tier}})

    def test_all_legs_ok_reads_green_with_a_pass_shaped_bar(self):
        # P = 13 -> ideal = 12; ratio = 0.10 sits inside [lower, upper] for
        # r = 0 -- same knob values `test_frontend_ab_dry_run.py`'s own
        # `test_default_knobs_produce_a_pass_shaped_bar` pins.
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_all_ok(raw_dir)
            report = frontend_ab_merge.build_report(
                Path(raw_dir), r=0.0, n=24, tip_sha="t" * 40, base_sha="b" * 40,
                box="test-box", dry_run=True,
            )
        self.assertEqual(report["status"], "GREEN", report)
        self.assertEqual(report["htsat_bar"]["verdict"], "PASS", report["htsat_bar"])
        self.assertEqual(report["htsat_bar"]["p"], 13)
        self.assertEqual(report["htsat_bar"]["ideal"], 12.0)

    def test_one_missing_leg_marks_the_report_invalid_with_no_bar(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            self._write_all_ok(raw_dir)
            # Overwrite one leg's .exit to a nonzero failure -- exactly
            # what a crashed real `finetune-run` invocation would leave.
            (Path(raw_dir) / "htsat__base__r1.exit").write_text("1")
            report = frontend_ab_merge.build_report(
                Path(raw_dir), r=0.0, n=24, tip_sha="t" * 40, base_sha="b" * 40,
                box="test-box", dry_run=True,
            )
        self.assertEqual(report["status"], "INVALID", report)
        self.assertIsNone(report["htsat_bar"])
        self.assertEqual(report["towers"]["htsat"]["base"]["r1"]["outcome"], "FAIL")


class MainEntryPointTests(unittest.TestCase):
    """`main(argv)` -- the exact call `frontend_ab.sh` makes -- writes the
    merged report to disk AND echoes it on stdout."""

    def test_main_writes_the_merged_report_to_out_path(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            for tower in frontend_ab_merge.TOWERS:
                for role in frontend_ab_merge.ROLES:
                    for repeat in frontend_ab_merge.REPEATS:
                        tier = {
                            "steps_measured": 100,
                            "media_front_end_wall_s": 1.0,
                            "train_run_wall_s": 15.0,
                        }
                        if role == "tip":
                            tier["rayon_pool_threads"] = 13
                        _write_leg(raw_dir, tower, role, repeat, {"tiers": {"finetune_run": tier}})
            out_path = os.path.join(out_dir, "report.json")
            rc = frontend_ab_merge.main(
                [raw_dir, out_path, "0.0", "24", "t" * 40, "b" * 40, "test-box", "0"]
            )
            self.assertEqual(rc, 0)
            with open(out_path, encoding="utf-8") as f:
                written = json.load(f)
            self.assertEqual(written["status"], "GREEN", written)
            self.assertFalse(written["dry_run"])


if __name__ == "__main__":
    unittest.main()
