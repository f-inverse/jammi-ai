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

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import howwell_dose_ladder_cause as namer  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
