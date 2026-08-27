#!/usr/bin/env python3
"""Tests for `check_lead_gate.py` — the totality cross-check pattern from
`test_check_ci_guard_wiring.py`: don't just trust that `check_lead_gate.py
--self-test` exists as a flag, prove it is actually WIRED into
`.github/workflows/swarm.yml`'s real `run:`/`cmd:` step body (comments do not
count — reuses `check_ci_guard_wiring.workflow_run_text()` rather than
reimplementing that parse), AND that running it actually drives the real
hook scripts and exits 0 in this checkout.

Run directly: `python3 ci/scripts/test_check_lead_gate.py`
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_ci_guard_wiring as cgw  # noqa: E402
import check_lead_gate as clg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


class SelfTestIsWiredIntoSwarmYml(unittest.TestCase):
    """The totality cross-check: `check_lead_gate.py --self-test` must
    appear in a real step body of `.github/workflows/swarm.yml`, not merely
    in a comment (the exact F7 shape `check_ci_guard_wiring.py`'s own module
    doc names — reused here, not reimplemented, so this test cannot silently
    drift from what that gate actually enforces)."""

    def test_self_test_flag_appears_in_a_run_step_not_a_comment(self):
        corpus = cgw.workflow_run_text()
        self.assertIn(
            "check_lead_gate.py --self-test",
            corpus,
            "the swarm workflow must invoke `check_lead_gate.py --self-test` from a "
            "real run:/cmd: step body — check_ci_guard_wiring.py's own comment-stripping "
            "parse is reused here so this assertion tracks the SAME definition of 'wired'",
        )

    def test_check_lead_gate_py_itself_is_wired(self):
        """`check_lead_gate.py` is a `check_*.py` script under `ci/scripts/`
        — `check_ci_guard_wiring.py`'s own `gate_scripts()` sweep already
        requires it to be wired; this test pins that this file specifically
        satisfies it (a fast, targeted echo of the general gate)."""
        corpus = cgw.workflow_run_text()
        self.assertIn("check_lead_gate.py", corpus)

    def test_test_check_lead_gate_py_itself_is_wired(self):
        """This very test suite is a `test_*.py` under `ci/` —
        `check_ci_guard_wiring.py`'s `tracked_test_suites()` sweep requires
        IT to be wired too; pinned here so a future PR cannot silently drop
        the workflow step and let the wiring gate be the only thing that
        notices."""
        corpus = cgw.workflow_run_text()
        self.assertIn("test_check_lead_gate.py", corpus)


class SelfTestActuallyExecutes(unittest.TestCase):
    """Drives the REAL `check_lead_gate.py --self-test` entry point as a
    subprocess (not `clg.self_test()` called in-process, so this exercises
    the exact command line CI runs) and asserts it exits 0 — i.e. that the
    self-test itself, which in turn spawns the real hook scripts, actually
    runs and passes in this checkout."""

    def test_self_test_subprocess_exits_zero(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "ci" / "scripts" / "check_lead_gate.py"), "--self-test"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode, 0,
            f"check_lead_gate.py --self-test must exit 0.\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )
        for name, _fn in clg.FIXTURES:
            self.assertIn(f"check-lead-gate[{name}]: OK", proc.stdout, f"fixture {name} did not report OK")


class NoUsageArgsIsAnError(unittest.TestCase):
    def test_main_without_self_test_flag_is_non_zero(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "ci" / "scripts" / "check_lead_gate.py")],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
