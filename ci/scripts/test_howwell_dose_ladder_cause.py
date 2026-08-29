#!/usr/bin/env python3
"""Hermetic `unittest` suite for `howwell_dose_ladder_cause.py` (unit-63
round-13 audit F1) -- drives the real `dose_ladder_cause` pure function
against in-memory synthetic `finetune_run_ab_report.json`-shaped dicts,
mirroring `test_check_kernel_oracles.py`'s own "drive the real entry points
against throwaway fixtures" shape for this repo's `test_*.py` gate-suite
convention.

Run: `python3 ci/scripts/test_howwell_dose_ladder_cause.py`
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "perf"))

import ab_merge  # noqa: E402
import howwell_dose_ladder_cause as namer  # noqa: E402

_SCRIPT = str(Path(__file__).resolve().parent / "howwell_dose_ladder_cause.py")


class DoseLadderCauseTests(unittest.TestCase):
    def test_red_proof_only_cause(self):
        # unit-63 round-13 audit F1's own named failure shape: primary
        # decision GREEN, RED-proof undischarged, no other dose-ladder
        # cause present. Pre-fix (74fd69ef), this fell through to the
        # "unknown" fallback -- the exact unexplained-contradiction shape
        # this namer exists to prevent.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "redproof-nobc", "detected": "not-detected"}],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", cause)
        self.assertNotIn("unknown", cause)

    def test_mixed_causes_all_named(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": "dose_label 'eps-bogus' failed to parse",
                "dose_anomalies": [{"dose_label": "eps-0.50"}],
                "doses": [
                    {"dose_label": "eps-0.10", "detected": "INVALID"},
                    {"dose_label": "redproof-nobc", "detected": "not-detected"},
                ],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("sensitivity_error", cause)
        self.assertIn("invalid_doses=eps-0.10", cause)
        self.assertIn("dose_anomalies", cause)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", cause)

    def test_proven_red_proof_never_named_as_a_cause(self):
        # PROVEN contributes nothing to ab_merge.py's own exit code (CONTRACT
        # F4) -- the namer must never name a PROVEN red_proof_verdict as a
        # GREEN-but-nonzero cause.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": "dose_label 'eps-bogus' failed to parse",
                "dose_anomalies": [],
                "doses": [],
                "red_proof_verdict": "PROVEN",
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("sensitivity_error", cause)
        self.assertNotIn("red_proof_verdict", cause)

    def test_all_clear_fallback_enumerates_all_four_causes(self):
        # unit-63 round-13 audit F1: the fallback text must name every
        # cause class this namer checked, not just the eps-family three --
        # a bare "unknown" (pre-fix) looks like this namer forgot to check
        # something, rather than affirmatively ruling all four out.
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "eps0.50", "detected": "RED"}],
                "red_proof_verdict": None,
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("unknown", cause)
        self.assertIn("dose_anomalies", cause)
        self.assertIn("sensitivity_error", cause)
        self.assertIn("invalid dose column", cause)
        self.assertIn("red_proof_verdict", cause)

    def test_no_mutant_dose_ladder_key_falls_back_cleanly(self):
        cause = namer.dose_ladder_cause({"status": "GREEN"})
        self.assertIn("unknown", cause)


class DoseLadderCauseNamesBoundToAbMergeExitFoldTests(unittest.TestCase):
    """Unit-63 round-14 audit F6: the namer's own checked-cause set must
    equal `ab_merge.py`'s own `main()` dose-ladder exit-fold cause set --
    imports BOTH modules and asserts equality, so a fifth cause added to one
    side without the other is a RED test here, never silent drift (the prior
    state: `_ALL_CAUSE_NAMES`'s own comment CLAIMED this with nothing
    mechanical enforcing it).
    """

    def test_namer_cause_names_equal_ab_merge_dose_ladder_exit_cause_names(self):
        self.assertEqual(set(namer._ALL_CAUSE_NAMES), set(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES))

    def test_namer_reuses_the_constant_directly_never_a_hand_duplicated_literal(self):
        # The strongest binding available: literally the same list contents,
        # imported from the one place `ab_merge.py`'s own exit fold is
        # itself asserted against (see that module's own `main()` doc).
        self.assertEqual(list(namer._ALL_CAUSE_NAMES), list(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES))

    def test_exactly_four_causes_today(self):
        # Pinned count -- a change here is a real fifth-cause addition (or a
        # removal), never an incidental refactor; re-derive, never bump to
        # make this test pass.
        self.assertEqual(len(ab_merge.DOSE_LADDER_EXIT_CAUSE_NAMES), 4)


class DosesFieldHardeningTests(unittest.TestCase):
    """Unit-63 round-14 audit A4: `ladder["doses"]` is a producer/merger
    artifact field, never assumed well-shaped -- `null`, a non-list value, or
    a list carrying a `null`/non-dict element must degrade to a NAMED cause,
    never an uncaught exception the shell's own `2>/dev/null || echo
    "unknown (could not inspect ...)"` fallback would silently swallow into
    an opaque, indistinguishable-from-"nothing wrong" "unknown".
    """

    def test_doses_field_absent_is_not_a_malformation(self):
        report = {"status": "GREEN", "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": []}}
        cause = namer.dose_ladder_cause(report)
        self.assertIn("unknown", cause)
        self.assertNotIn("malformed", cause)
        self.assertNotIn("doses_field", cause)

    def test_doses_field_null_degrades_to_a_named_cause_never_a_crash(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": None},
        }
        cause = namer.dose_ladder_cause(report)  # must not raise
        self.assertIn("doses_field_is_null", cause)

    def test_doses_field_not_a_list_degrades_to_a_named_cause(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": "not-a-list"},
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_is_not_a_list", cause)
        self.assertIn("type=str", cause)

    def test_doses_field_with_null_elements_degrades_to_a_named_cause_and_still_scans_the_rest(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [None, {"dose_label": "eps-0.10", "detected": "INVALID"}, "also-not-a-dict"],
            },
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_has_2_malformed_entries", cause)
        self.assertIn("invalid_doses=eps-0.10", cause)

    def test_doses_field_with_one_null_element_uses_singular_wording(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {"sensitivity_error": None, "dose_anomalies": [], "doses": [None]},
        }
        cause = namer.dose_ladder_cause(report)
        self.assertIn("doses_field_has_1_malformed_entry", cause)
        self.assertNotIn("entries", cause)


class MainEntryPointTests(unittest.TestCase):
    """Unit-63 round-14 audit A5: `main()` (the actual CLI entry
    `runpod_gpu_howwell.sh` invokes) had zero execution coverage -- every
    existing test drove `dose_ladder_cause` directly. Covers argv handling,
    a missing file, and a valid file, via BOTH a real subprocess invocation
    (the exact shape `runpod_gpu_howwell.sh` uses) and a direct `main()`
    call (for exit-code assertions without process-spawn overhead).
    """

    def test_wrong_argv_count_prints_usage_and_returns_2(self):
        self.assertEqual(namer.main([]), 2)
        self.assertEqual(namer.main(["a", "b"]), 2)

    def test_missing_file_subprocess_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, _SCRIPT, "/nonexistent/path/does-not-exist.json"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)

    def test_valid_file_subprocess_prints_cause_and_exits_zero(self):
        report = {
            "status": "GREEN",
            "mutant_dose_ladder": {
                "sensitivity_error": None,
                "dose_anomalies": [],
                "doses": [{"dose_label": "redproof-nobc", "detected": "not-detected"}],
                "red_proof_verdict": "NOT_PROVEN (redproof-nobc=not-detected)",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(report, fh)
            path = fh.name
        try:
            proc = subprocess.run([sys.executable, _SCRIPT, path], capture_output=True, text=True)
        finally:
            Path(path).unlink()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("red_proof_verdict=NOT_PROVEN (redproof-nobc=not-detected)", proc.stdout)

    def test_wrong_argv_count_subprocess_exits_2_with_usage_on_stderr(self):
        proc = subprocess.run([sys.executable, _SCRIPT], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage:", proc.stderr)

    def test_main_direct_call_matches_dose_ladder_cause_output(self):
        # A direct `main()` call over a real file, cross-checked against
        # `dose_ladder_cause` called directly on the same dict -- proves
        # `main()` is not a second, independently-drifting read path.
        report = {"status": "GREEN", "mutant_dose_ladder": {"sensitivity_error": "boom", "dose_anomalies": []}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump(report, fh)
            path = fh.name
        try:
            import io
            from contextlib import redirect_stdout

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = namer.main([path])
        finally:
            Path(path).unlink()
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue().strip(), namer.dose_ladder_cause(report))


if __name__ == "__main__":
    unittest.main()
