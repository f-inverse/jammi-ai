#!/usr/bin/env python3
"""Hermetic tests for `frontend_ab_artifact.py` (issue #421 follow-on
close-out artifact producer; contract `perf/421-frontend` v3).

Everything here is SYNTHETIC: a small `git init`'d tempdir repo (mirroring
`check_cuda_run_artifacts.py`'s own self-test discipline) stands in for
`--repo-root`, and every raw leg / `report.json` / identity sidecar is
hand-built in a tempdir -- no GPU, no pod, no network, and none of the real
pod-p421c measurement (that lives in the small COMMITTED fixture directory
`fixtures/frontend_ab_final/`, exercised separately by
`RealFixtureRegressionTests` below, never by this suite's own synthetic
classes).

`_oracle_bar` below is an INDEPENDENT re-implementation of the pinned bar
formula (a THIRD copy, after `frontend_ab_merge.py`'s own and
`frontend_ab_artifact.py`'s own) -- the numpy-first-oracle doctrine scaled
to plain-float arithmetic, per `profile_421_artifact.py`'s own test-suite
precedent ("the arithmetic here is simple enough that a second float
computation IS the independent check"). Every numeric field this suite
checks is checked against THIS oracle, never against a value copied out of
the module under test.

`LegSetEnumerationTests` drives the audit's own probe: a `--repeats`
shrink (3 -> 2) with the `r3` raw-leg files still present on disk (a stale
re-run) must REFUSE, never silently read only `r1`/`r2` and PASS -- in
both directions (extra legs, missing legs), on both `--raw-dir` and
`--report-json`'s own `towers` object independently. `RDomainTests`/
`RDomainEndToEndTests` cover `r`'s own `[0, 1)` domain at both edges (and
beyond), plus every division this module performs being guarded against a
zero/non-positive divisor (a named refusal, never a `ZeroDivisionError`).
`SerialTailFixtureTests` covers `--serial-tail`'s own missing-file/
missing-line/malformed-`t_s` refusals; `TowerTaskAndDeviceNameTests` covers
the per-leg `task`/`device_name` cross-check.

Run: `python3 ci/scripts/perf/test_frontend_ab_artifact.py`
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PERF_DIR = Path(__file__).resolve().parent
ARTIFACT = PERF_DIR / "frontend_ab_artifact.py"
REAL_FIXTURE_DIR = PERF_DIR / "fixtures" / "frontend_ab_final"

sys.path.insert(0, str(PERF_DIR))
import frontend_ab_artifact as art  # noqa: E402

TOWERS = ("htsat", "clip-vision")
ROLES = ("base", "tip")
REPEATS = ("r1", "r2", "r3")
N_ITEMS_PER_STEP = 24
P = 8
R_DRIVER = 0.0
MEASURED_SERIAL_TAIL_S = 0.05  # -> r_measured = 0.05 / front_base_mean_s(=1.0) = 0.05
BOX = "TESTBOX"
DEVICE_NAME = "TESTBOX"  # -- box.split(',')[0].strip(): the per-leg device_name cross-check
TASK_BY_TOWER = {"htsat": "audio_embedding", "clip-vision": "image_embedding"}

# front_per_step per (tower, role) -- identical across repeats (spread 0)
# keeps the oracle arithmetic trivial to hand-check; the "spread" family is
# covered separately below (SpreadTests).
FRONT_PER_STEP = {
    ("htsat", "base"): 1.0,
    ("htsat", "tip"): 0.2,
    ("clip-vision", "base"): 0.5,
    ("clip-vision", "tip"): 0.1,
}
TRAIN_PER_STEP = {
    ("htsat", "base"): 1.2,
    ("htsat", "tip"): 0.4,
    ("clip-vision", "base"): 0.6,
    ("clip-vision", "tip"): 0.2,
}
STEPS = 100


def _oracle_bar(front_base_vals: list[float], front_tip_vals: list[float], p: int, n: int, r: float) -> dict:
    """Independent re-implementation of the pinned bar formula -- never
    calls `frontend_ab_artifact.py` or `frontend_ab_merge.py`."""
    ideal = n / math.ceil(n / p)
    upper_bound = r + (1 - r) / (0.5 * ideal)
    lower_bound = r + (1 - r) / ideal
    front_tip = sum(front_tip_vals) / len(front_tip_vals)
    front_base = sum(front_base_vals) / len(front_base_vals)
    ratio = front_tip / front_base
    ratio_lo = min(front_tip_vals) / max(front_base_vals)
    ratio_hi = max(front_tip_vals) / min(front_base_vals)
    if ratio_hi <= upper_bound and ratio_lo >= lower_bound:
        verdict = "PASS"
    elif ratio_lo > upper_bound:
        verdict = "FAIL"
    elif ratio_hi < lower_bound:
        verdict = "INVALID_BEATS_IDEAL"
    else:
        verdict = "UNRESOLVED"
    return {
        "p": p, "n": n, "ideal": ideal, "r": r,
        "upper_bound": upper_bound, "lower_bound": lower_bound,
        "front_tip_mean_s": front_tip, "front_base_mean_s": front_base,
        "ratio": ratio, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# a tiny `git init`'d tempdir repo -- mirrors `check_cuda_run_artifacts.py`'s
# own self-test discipline. Two commits: `base_sha` (touches `unit/hot.rs`
# for the first time) and `tip_sha` (modifies it again) -- these two shas
# are what every fixture's own `base_sha`/`tip_sha` + every raw leg's own
# `provenance.build_sha` are set to, so `frontend_ab_artifact.py`'s git
# calls resolve against REAL commits, never a fabricated string.
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {
        "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args} failed: {proc.stderr}")
    return proc


def _init_repo(root: Path) -> tuple[str, str]:
    _git(["init", "-q"], root)
    (root / "unit").mkdir()
    (root / "unit" / "hot.rs").write_text("v0\n", encoding="utf-8")
    _git(["add", "."], root)
    _git(["commit", "-q", "-m", "base"], root)
    base_sha = _git(["rev-parse", "HEAD"], root).stdout.strip()

    (root / "unit" / "hot.rs").write_text("v1\n", encoding="utf-8")
    _git(["commit", "-q", "-am", "tip: touches the hot path"], root)
    tip_sha = _git(["rev-parse", "HEAD"], root).stdout.strip()
    return base_sha, tip_sha


def _write_raw_leg(
    raw_dir: Path, tower: str, role: str, repeat: str, *, build_sha: str,
    front_per_step: float, train_per_step: float, steps: int = STEPS,
    rayon_pool_threads: int | None = None, exit_code: int = 0, report_override: dict | None = None,
    task: str | None = None, device_name: str = DEVICE_NAME,
) -> None:
    stem = f"{tower}__{role}__{repeat}"
    (raw_dir / f"{stem}.exit").write_text(str(exit_code), encoding="utf-8")
    (raw_dir / f"{stem}.stderr").write_text("", encoding="utf-8")
    if report_override is not None:
        report = report_override
    else:
        tier = {
            "steps_measured": steps,
            "media_front_end_wall_s": front_per_step * steps,
            "train_run_wall_s": train_per_step * steps,
            "task": task if task is not None else TASK_BY_TOWER[tower],
            "device_name": device_name,
        }
        if rayon_pool_threads is not None:
            tier["rayon_pool_threads"] = rayon_pool_threads
        report = {
            "host": {"logical_cpus": rayon_pool_threads or 4},
            "provenance": {"build_sha": build_sha, "target": "x86_64-unknown-linux-gnu", "profile": "release"},
            "tiers": {"finetune_run": tier},
        }
    (raw_dir / f"{stem}.json").write_text(json.dumps(report), encoding="utf-8")


def _write_serial_tail(root: Path, *, audio_t_s: float | str | None = MEASURED_SERIAL_TAIL_S, extra_lines: list[str] | None = None) -> Path:
    """A synthetic stand-in for the committed `fixtures/frontend_ab_final/
    serial_tail.txt` -- the same 'task=... t_s=...' line shape, never the
    real file (that is exercised separately by `RealFixtureRegressionTests`
    below). `audio_t_s=None` omits the audio_embedding line entirely (for
    the 'missing line' refusal test); a string value writes it verbatim
    (for the 'malformed t_s' refusal test)."""
    lines = []
    if audio_t_s is not None:
        lines.append(f"task=audio_embedding device=cuda:0 reps=20 dims=[24, 4, 1001, 64] t_s={audio_t_s}")
    lines.append("task=image_embedding device=cuda:0 reps=20 dims=[24, 3, 224, 224] t_s=0.0011608887")
    if extra_lines:
        lines.extend(extra_lines)
    path = root / "serial_tail.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_fixture(root: Path, base_sha: str, tip_sha: str, *, repeats: tuple[str, ...] = REPEATS) -> dict:
    raw_dir = root / "raw"
    raw_dir.mkdir()
    for tower in TOWERS:
        for role in ROLES:
            sha = base_sha if role == "base" else tip_sha
            for repeat in repeats:
                _write_raw_leg(
                    raw_dir, tower, role, repeat, build_sha=sha,
                    front_per_step=FRONT_PER_STEP[(tower, role)],
                    train_per_step=TRAIN_PER_STEP[(tower, role)],
                    rayon_pool_threads=P if role == "tip" else None,
                )

    towers_json = {
        tower: {
            role: {
                repeat: {
                    "outcome": "OK",
                    "steps_measured": STEPS,
                    "front_per_step": FRONT_PER_STEP[(tower, role)],
                    "train_per_step": TRAIN_PER_STEP[(tower, role)],
                    "rayon_pool_threads": P if role == "tip" else None,
                }
                for repeat in repeats
            }
            for role in ROLES
        }
        for tower in TOWERS
    }
    htsat_bar = _oracle_bar(
        [FRONT_PER_STEP[("htsat", "base")]] * len(repeats),
        [FRONT_PER_STEP[("htsat", "tip")]] * len(repeats),
        P, N_ITEMS_PER_STEP, R_DRIVER,
    )
    cv_front_tip = FRONT_PER_STEP[("clip-vision", "tip")]
    cv_front_base = FRONT_PER_STEP[("clip-vision", "base")]
    report_json = {
        "tool": "frontend_ab.sh",
        "dry_run": False,
        "base_sha": base_sha,
        "tip_sha": tip_sha,
        "box": BOX,
        "serial_tail_ratio": R_DRIVER,
        "n_items_per_step": N_ITEMS_PER_STEP,
        "repeats": len(repeats),
        "status": "GREEN",
        "towers": towers_json,
        "htsat_bar": htsat_bar,
        "clip_vision_report_only": {
            "front_tip_mean_s": cv_front_tip,
            "front_base_mean_s": cv_front_base,
            "ratio": cv_front_tip / cv_front_base,
        },
    }
    identity = {
        "what": "fixture identity",
        "gpu": "FIXTURE-GPU",
        "driver": "0.0.0",
        "cpu": "FIXTURE-CPU",
        "recorded_deviations": ["fixture deviation"],
        "measured_tip_precedes_merge_tip_commentary": "fixture commentary",
    }
    report_path = root / "report.json"
    identity_path = root / "identity.json"
    report_path.write_text(json.dumps(report_json), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    serial_tail_path = _write_serial_tail(root)
    return {
        "raw_dir": raw_dir,
        "report_path": report_path,
        "report_json": report_json,
        "identity_path": identity_path,
        "identity": identity,
        "serial_tail_path": serial_tail_path,
        "htsat_bar_oracle": htsat_bar,
    }


class HappyPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, serial_tail_path=None):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"],
            serial_tail_path if serial_tail_path is not None else self.fixture["serial_tail_path"], self.root,
            "python3 ci/scripts/perf/frontend_ab_artifact.py (test invocation)",
        )

    def test_top_level_shape(self):
        report = self._build()
        self.assertEqual(set(report.keys()), {"schema_version", "git_sha", "box", "producer", "status", "notes", "measurement", "verdict"})
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["git_sha"], self.tip_sha)
        self.assertEqual(report["box"], "TESTBOX")
        self.assertEqual(report["producer"]["path"], "ci/scripts/perf/frontend_ab_artifact.py")
        self.assertEqual(report["producer"]["kind"], "script")
        self.assertEqual(report["producer"]["gating"], "none")
        self.assertEqual(report["status"], "GREEN")

    def test_htsat_bar_driver_matches_independent_oracle(self):
        report = self._build()
        oracle = _oracle_bar(
            [FRONT_PER_STEP[("htsat", "base")]] * 3, [FRONT_PER_STEP[("htsat", "tip")]] * 3, P, N_ITEMS_PER_STEP, R_DRIVER
        )
        got = report["measurement"]["htsat_bar_driver_r"]
        for field, expected in oracle.items():
            if isinstance(expected, float):
                self.assertAlmostEqual(got[field], expected, places=9, msg=field)
            else:
                self.assertEqual(got[field], expected, msg=field)
        self.assertEqual(oracle["verdict"], "PASS")  # sanity: this fixture is a PASS scenario

    def test_htsat_bar_measured_matches_independent_oracle(self):
        report = self._build()
        r_measured = MEASURED_SERIAL_TAIL_S / FRONT_PER_STEP[("htsat", "base")]
        oracle = _oracle_bar(
            [FRONT_PER_STEP[("htsat", "base")]] * 3, [FRONT_PER_STEP[("htsat", "tip")]] * 3, P, N_ITEMS_PER_STEP, r_measured
        )
        got = report["measurement"]["htsat_bar_measured_r"]
        for field, expected in oracle.items():
            if isinstance(expected, float):
                self.assertAlmostEqual(got[field], expected, places=9, msg=field)
            else:
                self.assertEqual(got[field], expected, msg=field)

    def test_verdict_block(self):
        report = self._build()
        v = report["verdict"]
        self.assertEqual(v["unit_verdict"], "PASS")
        self.assertEqual(v["cross_checked_against_report_json_htsat_bar_verdict"], "PASS")
        dev = v["serial_tail_ratio_deviation"]
        self.assertEqual(dev["kind"], "driver default vs measured; verdict invariant")
        self.assertAlmostEqual(dev["r_driver"], R_DRIVER, places=12)
        self.assertAlmostEqual(dev["r_measured"], MEASURED_SERIAL_TAIL_S / FRONT_PER_STEP[("htsat", "base")], places=9)
        self.assertEqual(dev["verdict_under_r_driver"], "PASS")
        self.assertEqual(dev["verdict_under_r_measured"], "PASS")
        self.assertTrue(dev["invariant_holds"])
        self.assertIn("ACTIVATE", v["contract_clause_applied"])

    def test_clip_vision_report_only_matches_oracle(self):
        report = self._build()
        cv = report["measurement"]["clip_vision_report_only"]
        expected_ratio = FRONT_PER_STEP[("clip-vision", "tip")] / FRONT_PER_STEP[("clip-vision", "base")]
        self.assertAlmostEqual(cv["ratio"], expected_ratio, places=9)
        self.assertAlmostEqual(cv["front_tip_mean_s"], FRONT_PER_STEP[("clip-vision", "tip")], places=9)
        self.assertAlmostEqual(cv["front_base_mean_s"], FRONT_PER_STEP[("clip-vision", "base")], places=9)

    def test_notes_carry_git_facts_and_hashes(self):
        report = self._build()
        notes = report["notes"]
        self.assertEqual(notes["driver_commit"], self.tip_sha)
        self.assertEqual(notes["rendered_from_tree_sha"], self.tip_sha)  # HEAD never moved past tip in this fixture
        self.assertIn("report_json", notes["input_sha256"])
        self.assertIn("identity", notes["input_sha256"])
        self.assertIn("serial_tail", notes["input_sha256"])
        self.assertEqual(len(notes["input_sha256"]), 3 + len(TOWERS) * len(ROLES) * len(REPEATS))  # report+identity+serial_tail + every raw leg
        self.assertTrue(all(len(v) == 64 for v in notes["input_sha256"].values()))
        self.assertEqual(len(notes["run_sha256"]), 64)
        self.assertIn("fixture deviation", notes["recorded_deviations"])
        dev = notes["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["kind"], "measured-tip-precedes-merge-tip")
        self.assertEqual(dev["measured_tip"], self.tip_sha)
        self.assertEqual(dev["rendered_from_tree_sha"], self.tip_sha)  # HEAD never moved past tip in this fixture
        self.assertEqual(dev["files_changed_since_measured_tip"], [])
        self.assertEqual(dev["commentary"], "fixture commentary")

    def test_legs_present_for_every_tower_role_repeat(self):
        report = self._build()
        legs = report["measurement"]["legs"]
        self.assertEqual(len(legs), 4 * 3)
        for tower in TOWERS:
            for role in ROLES:
                for repeat in REPEATS:
                    key = f"{tower}__{role}__{repeat}"
                    self.assertIn(key, legs)
                    self.assertAlmostEqual(legs[key]["front_per_step"], FRONT_PER_STEP[(tower, role)], places=9)

    def test_regeneration_is_byte_identical(self):
        payload_a = json.dumps(self._build(), sort_keys=False)
        payload_b = json.dumps(self._build(), sort_keys=False)
        self.assertEqual(payload_a, payload_b)


class MeasuredTipPrecedesMergeTipTests(unittest.TestCase):
    """Files changed AFTER the measured tip are RECORDED mechanically
    (never refused: a branch tip that was "measured" can legitimately gain
    more commits before it merges -- exactly what happened on
    `perf/421-frontend` itself, a concurrent fold landing while this very
    artifact was being built), with the reviewed `--identity` commentary
    attached alongside the mechanical file list, never replacing it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"], self.fixture["serial_tail_path"], self.root, "inv",
        )

    def test_post_tip_change_is_recorded_mechanically_with_reviewed_commentary(self):
        (self.root / "docs").mkdir()
        (self.root / "docs" / "other.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "docs/other.md"], self.root)  # NEVER "add ." here -- the fixture's own raw_dir/report.json/
        # identity.json already sit untracked in this same tempdir (see _write_fixture) and would otherwise be
        # swept into this commit too.
        _git(["commit", "-q", "-m", "post-tip change"], self.root)
        head = _git(["rev-parse", "HEAD"], self.root).stdout.strip()

        report = self._build()
        self.assertEqual(report["notes"]["rendered_from_tree_sha"], head)
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["kind"], "measured-tip-precedes-merge-tip")
        self.assertEqual(dev["measured_tip"], self.tip_sha)
        self.assertEqual(dev["rendered_from_tree_sha"], head)
        self.assertEqual(dev["files_changed_since_measured_tip"], ["docs/other.md"])
        self.assertIn("docs/other.md", dev["diffstat"])
        self.assertEqual(dev["commentary"], "fixture commentary")

    def test_post_tip_change_that_touches_this_units_own_files_is_still_only_recorded(self):
        # A file-diff alone cannot judge whether an edit changed timed
        # behaviour -- even a change to the SAME file this unit's own scope
        # touched (`unit/hot.rs`) is recorded, never refused; that judgement
        # lives in --identity's own reviewed commentary, not in this
        # mechanism.
        (self.root / "unit" / "hot.rs").write_text("v2 -- touched again after the measured tip\n", encoding="utf-8")
        _git(["commit", "-q", "-am", "post-tip, touches the same file again"], self.root)

        report = self._build()
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["files_changed_since_measured_tip"], ["unit/hot.rs"])

    def test_no_post_tip_changes_is_recorded_as_an_empty_list(self):
        report = self._build()
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["files_changed_since_measured_tip"], [])
        self.assertEqual(dev["diffstat"], "")

    def test_shallow_repo_is_refused(self):
        with mock.patch.object(art, "_is_shallow_repository", return_value=True):
            with self.assertRaises(art.ArtifactBuildError) as ctx:
                self._build()
        self.assertIn("shallow checkout", str(ctx.exception))


class RefusalTests(unittest.TestCase):
    """Every way the input can be missing, failed, mismatched, or non-finite
    must refuse -- never silently pick a value or propagate a NaN."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture, serial_tail_path=None):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"],
            serial_tail_path if serial_tail_path is not None else fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_missing_exit_file(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__base__r1.exit").unlink()
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("missing .exit file", str(ctx.exception))

    def test_refuses_nonzero_exit(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__tip__r2.exit").write_text("3", encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("leg exited 3", str(ctx.exception))

    def test_refuses_build_sha_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r1.json").read_text())
        bad["provenance"]["build_sha"] = "f" * 40
        (fixture["raw_dir"] / "htsat__base__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not match the base sha", str(ctx.exception))

    def test_refuses_nonfinite_media_front_end_wall_s(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "clip-vision__tip__r1.json").read_text())
        bad["tiers"]["finetune_run"]["media_front_end_wall_s"] = math.nan
        (fixture["raw_dir"] / "clip-vision__tip__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_infinite_train_run_wall_s(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r3.json").read_text())
        bad["tiers"]["finetune_run"]["train_run_wall_s"] = math.inf
        (fixture["raw_dir"] / "htsat__base__r3.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_front_per_step_disagreement_with_report_json(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["towers"]["htsat"]["tip"]["r1"]["front_per_step"] = 999.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("front_per_step", str(ctx.exception))

    def test_refuses_htsat_bar_verdict_disagreement(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["htsat_bar"]["verdict"] = "FAIL"
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagrees with --report-json", str(ctx.exception))

    def test_refuses_rayon_pool_threads_disagreement_across_tip_legs(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__tip__r2.json").read_text())
        bad["tiers"]["finetune_run"]["rayon_pool_threads"] = 999
        (fixture["raw_dir"] / "htsat__tip__r2.json").write_text(json.dumps(bad), encoding="utf-8")
        fixture["report_json"]["towers"]["htsat"]["tip"]["r2"]["rayon_pool_threads"] = 999
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagree on rayon_pool_threads", str(ctx.exception))

    def test_refuses_base_leg_with_unexpected_rayon_pool_threads(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r1.json").read_text())
        bad["tiers"]["finetune_run"]["rayon_pool_threads"] = 8
        (fixture["raw_dir"] / "htsat__base__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("unexpectedly carries rayon_pool_threads", str(ctx.exception))

    def test_refuses_verdict_invariant_violation(self):
        # A measured serial tail large enough to push r_measured's own
        # verdict off PASS while r_driver=0.0 stays PASS -- the "verdict
        # invariant" this contract names must be a checked equality, not
        # merely asserted prose.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad_serial_tail = _write_serial_tail(self.root, audio_t_s=0.5)  # r_measured = 0.5 -> way past PASS
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture, serial_tail_path=bad_serial_tail)
        self.assertIn("verdict invariant", str(ctx.exception))

    def test_refuses_missing_towers_key(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        del fixture["report_json"]["towers"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("missing required key 'towers'", str(ctx.exception))

    def test_refuses_leg_outcome_not_ok(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["towers"]["clip-vision"]["base"]["r1"]["outcome"] = "FAIL"
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("own outcome is", str(ctx.exception))


class LegSetEnumerationTests(unittest.TestCase):
    """`_validate_leg_set`/`_validate_towers_repeat_keys` -- the leg set on
    `--raw-dir` and `--report-json`'s own `towers` object must equal
    EXACTLY `TOWERS x ROLES x {r1..r<repeats>}`, in BOTH directions. This
    covers the audit's own probe: `--repeats` shrinking from 3 to 2 while
    the `r3` raw-leg files are STILL PRESENT on disk (e.g. a stale re-run
    left them behind) must REFUSE, never silently read only r1/r2 and
    PASS."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"], fixture["serial_tail_path"], self.root, "inv",
        )

    def test_extra_leg_on_disk_beyond_declared_repeats_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha, repeats=("r1", "r2", "r3"))
        fixture["report_json"]["repeats"] = 2
        for tower in TOWERS:
            for role in ROLES:
                del fixture["report_json"]["towers"][tower][role]["r3"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not carry exactly", str(ctx.exception))
        self.assertIn("extra leg stems", str(ctx.exception))

    def test_missing_leg_on_disk_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__tip__r3.json").unlink()
        (fixture["raw_dir"] / "htsat__tip__r3.exit").unlink()
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not carry exactly", str(ctx.exception))
        self.assertIn("missing leg stems", str(ctx.exception))

    def test_stray_extra_file_of_an_unknown_tower_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "unknown-tower__base__r1.json").write_text("{}", encoding="utf-8")
        (fixture["raw_dir"] / "unknown-tower__base__r1.exit").write_text("0", encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("extra leg stems", str(ctx.exception))

    def test_report_json_towers_with_extra_repeat_key_is_refused(self):
        # --raw-dir itself carries EXACTLY the repeats=2 leg set, but
        # --report-json's own towers.htsat.tip object still carries a
        # leftover 'r3' entry -- independent of whatever sits on disk.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha, repeats=("r1", "r2"))
        fixture["report_json"]["towers"]["htsat"]["tip"]["r3"] = dict(
            fixture["report_json"]["towers"]["htsat"]["tip"]["r1"]
        )
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("towers.htsat.tip", str(ctx.exception))

    def test_report_json_towers_missing_a_repeat_key_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        del fixture["report_json"]["towers"]["clip-vision"]["base"]["r2"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("towers.clip-vision.base", str(ctx.exception))


class TowerTaskAndDeviceNameTests(unittest.TestCase):
    """Advisory cross-check: every leg's own `tiers.finetune_run.task`/
    `.device_name` must match the tower being read / `--report-json`'s own
    `box` device prefix -- refusing by name on either mismatch, rather than
    trusting the filename alone."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"], fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_task_tower_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "htsat__base__r1.json"
        report = json.loads(raw_path.read_text())
        report["tiers"]["finetune_run"]["task"] = "image_embedding"
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("tiers.finetune_run.task", str(ctx.exception))

    def test_refuses_device_name_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "clip-vision__tip__r2.json"
        report = json.loads(raw_path.read_text())
        report["tiers"]["finetune_run"]["device_name"] = "SOME OTHER GPU"
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("device_name", str(ctx.exception))

    def test_refuses_missing_task_field(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "htsat__tip__r3.json"
        report = json.loads(raw_path.read_text())
        del report["tiers"]["finetune_run"]["task"]
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("carries no tiers.finetune_run.task", str(ctx.exception))

    def test_refuses_missing_device_name_field(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "clip-vision__base__r1.json"
        report = json.loads(raw_path.read_text())
        del report["tiers"]["finetune_run"]["device_name"]
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("carries no tiers.finetune_run.device_name", str(ctx.exception))


class SerialTailFixtureTests(unittest.TestCase):
    """`_read_measured_serial_tail_s` -- named refusal if `--serial-tail`
    cannot be read, carries no `task=audio_embedding ... t_s=...` line, or
    that line's own `t_s` does not parse as a float."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, serial_tail_path):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"], serial_tail_path, self.root, "inv",
        )

    def test_refuses_missing_serial_tail_file(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(self.root / "does-not-exist.txt")
        self.assertIn("could not read --serial-tail", str(ctx.exception))

    def test_refuses_missing_audio_task_line(self):
        bad = _write_serial_tail(self.root, audio_t_s=None)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(bad)
        self.assertIn("no 'task=audio_embedding ... t_s=...' line found", str(ctx.exception))

    def test_refuses_malformed_t_s(self):
        bad = _write_serial_tail(self.root, audio_t_s="not-a-float")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(bad)
        self.assertIn("does not parse as a float", str(ctx.exception))


class RDomainTests(unittest.TestCase):
    """`r` (a serial-tail/front-end time ratio) is only defined on
    `[0, 1)` -- both edges, and beyond, are named refusals. Boundary
    matrix: r=0 (valid), r just under 1 (valid), r=1 (refused), r=1.5
    (refused), r=-0.5 (refused). Also: any zero/non-positive front-end
    time used as a divisor (`front_base_mean_s`==0) is a named
    `ArtifactBuildError`, never a `ZeroDivisionError`."""

    def test_validate_r_accepts_zero(self):
        self.assertEqual(art._validate_r(0.0, "r"), 0.0)

    def test_validate_r_accepts_just_under_one(self):
        self.assertAlmostEqual(art._validate_r(0.999999, "r"), 0.999999)

    def test_validate_r_refuses_exactly_one(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art._validate_r(1.0, "r")
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_validate_r_refuses_above_one(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._validate_r(1.5, "r")

    def test_validate_r_refuses_negative(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._validate_r(-0.5, "r")

    def test_require_positive_divisor_refuses_zero(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art._require_positive_divisor(0.0, "front_base")
        self.assertIn("must be a finite positive number to divide by", str(ctx.exception))

    def test_require_positive_divisor_refuses_negative(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._require_positive_divisor(-1.0, "front_base")

    def test_require_positive_divisor_accepts_positive(self):
        self.assertEqual(art._require_positive_divisor(0.5, "front_base"), 0.5)


class RDomainEndToEndTests(unittest.TestCase):
    """The same boundary matrix as `RDomainTests`, driven end to end
    through `build_report` -- proving the domain guard is actually wired
    into the pipeline, not just a helper nothing calls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture, serial_tail_path=None):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"],
            serial_tail_path if serial_tail_path is not None else fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_r_driver_equal_to_one(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = 1.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_driver_above_one(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = 1.5
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_driver_negative(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = -0.5
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_measured_above_one(self):
        # front_base_mean_s == 1.0 in this fixture's own HTSAT base legs,
        # so a measured t_s of 1.5 -> r_measured = 1.5.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad_serial_tail = _write_serial_tail(self.root, audio_t_s=1.5)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture, serial_tail_path=bad_serial_tail)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_zero_htsat_front_base_is_a_named_refusal_not_zero_division(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        for repeat in REPEATS:
            raw_path = fixture["raw_dir"] / f"htsat__base__{repeat}.json"
            report = json.loads(raw_path.read_text())
            report["tiers"]["finetune_run"]["media_front_end_wall_s"] = 0.0
            raw_path.write_text(json.dumps(report), encoding="utf-8")
            fixture["report_json"]["towers"]["htsat"]["base"][repeat]["front_per_step"] = 0.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be a finite positive number to divide by", str(ctx.exception))


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_cli(self, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, str(ARTIFACT),
                "--raw-dir", str(self.fixture["raw_dir"]),
                "--report-json", str(self.fixture["report_path"]),
                "--identity", str(self.fixture["identity_path"]),
                "--serial-tail", str(self.fixture["serial_tail_path"]),
                "--repo-root", str(self.root),
                *extra_args,
            ],
            capture_output=True, text=True, timeout=120,
        )

    def test_cli_writes_a_schema_shaped_artifact(self):
        out_path = self.root / "artifact.json"
        result = self._run_cli("--out", str(out_path))
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        report = json.loads(out_path.read_text())
        self.assertEqual(report["git_sha"], self.tip_sha)
        self.assertEqual(report["verdict"]["unit_verdict"], "PASS")
        self.assertIn("frontend_ab_artifact: git_sha=", result.stderr)

    def test_cli_refuses_and_exits_nonzero_on_bad_input(self):
        (self.fixture["raw_dir"] / "htsat__tip__r1.exit").write_text("7", encoding="utf-8")
        result = self._run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("leg exited 7", result.stderr)


class RealFixtureRegressionTests(unittest.TestCase):
    """Drives the REAL, committed pod-p421c fixture (`fixtures/
    frontend_ab_final/`) through `build_report`, against the ACTUAL current
    checkout as `--repo-root` -- a regression guard that the committed
    `report.json` + raw legs still cross-check cleanly against this module's
    own re-derivation.

    Locally (or on any OTHER CI leg), this SKIPS -- never fails -- if the
    real fixture directory is absent, or if `--repo-root`'s own git history
    does not contain the fixture's `tip_sha`/`base_sha` (a shallow checkout
    of an unrelated repo running this file in isolation is not this test's
    concern). But `ci.yml`'s own `frontend_ab_artifact suite` matrix entry
    sets `JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED=1` (this repo's own
    zero-execution-is-RED doctrine, the same precedent
    `test_convert_legacy_bert_checkpoint.py::CiExecutionAssertionTests`
    already establishes for `JAMMI_CI_SAFETENSORS_EXPECTED`) AND
    `fetch_depth: "0"` -- under that env var, BOTH escape hatches turn into
    a hard `fail()` instead of a silent skip, so a regression that deletes
    the fixture directory, or a matrix-entry edit that drops `fetch_depth:
    "0"` and reintroduces a shallow checkout, shows up as a RED leg rather
    than an indistinguishable quiet skip."""

    _CI_ENV_VAR = "JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED"

    def _ci_expects_real_fixture(self) -> bool:
        return os.environ.get(self._CI_ENV_VAR) == "1"

    def _skip_or_fail(self, message: str) -> None:
        if self._ci_expects_real_fixture():
            self.fail(f"{message} -- but {self._CI_ENV_VAR}=1 (ci.yml's own matrix entry expects this to run)")
        self.skipTest(message)

    def test_real_fixture_cross_checks_cleanly(self):
        if not REAL_FIXTURE_DIR.is_dir():
            self._skip_or_fail(f"{REAL_FIXTURE_DIR} not present")
            return
        report_path = REAL_FIXTURE_DIR / "report.json"
        identity_path = PERF_DIR / "frontend_ab_final_identity.json"
        raw_dir = REAL_FIXTURE_DIR / "raw"
        serial_tail_path = REAL_FIXTURE_DIR / "serial_tail.txt"
        report_json = json.loads(report_path.read_text())
        identity = json.loads(identity_path.read_text())
        repo_root = PERF_DIR.parents[2]

        def _has_commit(sha: str) -> bool:
            proc = subprocess.run(["git", "cat-file", "-e", sha], cwd=repo_root, capture_output=True, text=True)
            return proc.returncode == 0

        if not (_has_commit(report_json["base_sha"]) and _has_commit(report_json["tip_sha"])):
            self._skip_or_fail(
                "this checkout's history does not contain the real fixture's base_sha/tip_sha "
                "(a shallow checkout -- fetch_depth: '0' is required for this matrix entry)"
            )
            return

        report = art.build_report(
            raw_dir, report_path, report_json, identity_path, identity,
            serial_tail_path, repo_root, "python3 ci/scripts/perf/frontend_ab_artifact.py (regression test)",
        )
        self.assertEqual(report["git_sha"], report_json["tip_sha"])
        self.assertEqual(report["verdict"]["unit_verdict"], "UNRESOLVED")
        self.assertEqual(report["measurement"]["htsat_bar_driver_r"]["verdict"], "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
