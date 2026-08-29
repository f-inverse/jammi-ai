#!/usr/bin/env python3
"""Tests for `check_ship_register.py` — the totality cross-check pattern from
`test_check_ci_guard_wiring.py`/`test_check_lead_gate.py`: don't just trust
that `--self-test` exists as a flag, prove it is actually WIRED into
`.github/workflows/swarm.yml`'s real `run:`/`cmd:` step body (reuses
`check_ci_guard_wiring.workflow_run_text()` rather than reimplementing that
parse), AND that running it actually drives the real gate logic and exits 0
in this checkout.

Run directly: `python3 ci/scripts/test_check_ship_register.py`
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_ci_guard_wiring as cgw  # noqa: E402
import check_ship_register as csr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


class SelfTestIsWiredIntoSwarmYml(unittest.TestCase):
    """The totality cross-check: `check_ship_register.py --self-test` must
    appear in a real step body of `.github/workflows/swarm.yml`, not merely
    in a comment (the exact F7 shape `check_ci_guard_wiring.py`'s own module
    doc names — reused here, not reimplemented, so this test cannot silently
    drift from what that gate actually enforces)."""

    def test_self_test_flag_appears_in_a_run_step_not_a_comment(self):
        corpus = cgw.workflow_run_text()
        self.assertIn(
            "check_ship_register.py --self-test",
            corpus,
            "the swarm workflow must invoke `check_ship_register.py --self-test` from a "
            "real run:/cmd: step body — check_ci_guard_wiring.py's own comment-stripping "
            "parse is reused here so this assertion tracks the SAME definition of 'wired'",
        )

    def test_check_ship_register_py_itself_is_wired(self):
        """`check_ship_register.py` is a `check_*.py` script under
        `ci/scripts/` — `check_ci_guard_wiring.py`'s own `gate_scripts()`
        sweep already requires it to be wired; this test pins that this file
        specifically satisfies it (a fast, targeted echo of the general
        gate)."""
        corpus = cgw.workflow_run_text()
        self.assertIn("check_ship_register.py", corpus)

    def test_test_check_ship_register_py_itself_is_wired(self):
        """This very test suite is a `test_*.py` under `ci/` —
        `check_ci_guard_wiring.py`'s `tracked_test_suites()` sweep requires
        IT to be wired too; pinned here so a future PR cannot silently drop
        the workflow step and let the wiring gate be the only thing that
        notices."""
        corpus = cgw.workflow_run_text()
        self.assertIn("test_check_ship_register.py", corpus)

    def test_register_touched_guard_step_present(self):
        corpus = cgw.workflow_run_text()
        self.assertIn("REGISTER_TOUCHED", corpus)

    def test_governance_touched_guard_step_present(self):
        corpus = cgw.workflow_run_text()
        self.assertIn("GOVERNANCE_TOUCHED", corpus)


class SelfTestActuallyExecutes(unittest.TestCase):
    """Drives the REAL `check_ship_register.py --self-test` entry point as a
    subprocess (not `csr.self_test()` called in-process, so this exercises
    the exact command line CI runs) and asserts it exits 0 and reports every
    fixture OK."""

    def test_self_test_subprocess_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "ci" / "scripts" / "check_ship_register.py"), "--self-test"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"check_ship_register.py --self-test must exit 0.\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        for name, _fn in csr.FIXTURES:
            self.assertIn(f"check-ship-register[{name}]: OK", proc.stdout, f"fixture {name} did not report OK")

    def test_self_test_covers_all_six_golden_escape_fixtures(self):
        """Fixtures (2)/(3)/(4)/(6) — and (1)/(5) — cite their escape/unit
        precedent ids in-file; a future edit that drops one silently would
        shrink the golden-escape coverage without any test noticing. Pin
        the count and the ids each fixture's docstring/name carries."""
        names = {name for name, _fn in csr.FIXTURES}
        for expected in (
            "golden-1-standing-live-block",
            "golden-2-esc064-residual-mismatch",
            "golden-3-esc063-empty-scan",
            "golden-4-esc066-residual-coverage",
            "golden-5-unit62-seeded-open",
            "golden-6-esc066-trigger",
        ):
            self.assertIn(expected, names, f"required-RED golden fixture {expected!r} missing")


class CiModeRunsCleanOnCurrentTree(unittest.TestCase):
    """No registers are seeded yet in this repo, so CI mode must exit 0
    (nothing to check is not an error)."""

    def test_ci_mode_subprocess_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "ci" / "scripts" / "check_ship_register.py")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"check_ship_register.py (CI mode) must exit 0 on the current tree.\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}",
        )


class MarkerFileAssertion(unittest.TestCase):
    """REPO_ROOT resolves with the Cargo.toml marker-file assertion (the
    esc-063 resolution pattern) — pinned here so a future refactor that
    changes the module's file depth cannot silently reintroduce the
    zero-files-scanned-reports-PASS shape."""

    def test_repo_root_carries_cargo_toml(self):
        self.assertTrue((csr.REPO_ROOT / "Cargo.toml").is_file())

    def test_repo_root_resolves_to_the_real_repo_root(self):
        self.assertEqual(csr.REPO_ROOT, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
