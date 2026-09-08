#!/usr/bin/env python3
"""Tests for `profile_421_render_docs.py` (#421 close-out, docs-ci fold).

Every test builds a THROWAWAY synthetic artifact + doc pair and monkeypatches
the module's `ARTIFACT`/`BLOCKS` onto them, so these tests are independent of
this repo's own real close-out artifact (which is frozen evidence, never
edited to make a test convenient).

Run directly: `python3 ci/scripts/perf/test_profile_421_render_docs.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_421_render_docs as r  # noqa: E402


def _synthetic_artifact() -> dict:
    return {
        "legs": [
            {
                "leg_id": "clip-text-A1",
                "tower": "clip-text",
                "dtype": "f32",
                "arm": "A",
                "kernels_disabled": [],
                "per_step": {
                    "wall_s_per_step": 0.1004,
                    "front_s_per_step": 0.0,
                    "busy_s_per_step": 0.0608,
                    "residual_s_per_step": 0.0396,
                    "front_share_of_wall": 0.0,
                    "busy_share_of_wall": 0.605,
                },
            },
            {
                "leg_id": "clip-text-D2",
                "tower": "clip-text",
                "dtype": "f32",
                "arm": "D",
                "kernels_disabled": ["layer_norm_fused", "lora_linear_fused"],
                "per_step": {
                    "wall_s_per_step": 0.1486,
                    "front_s_per_step": 0.0,
                    "busy_s_per_step": 0.0895,
                    "residual_s_per_step": 0.0591,
                    "front_share_of_wall": 0.0,
                    "busy_share_of_wall": 0.602,
                },
            },
        ],
        "candidate_decisions": [
            {"port": "C-ATTN-clip-text", "verdict": "UNRESOLVED", "reason": "a short reason"},
        ],
        "findings": [
            {"id": "a-finding", "text": "a short finding text"},
        ],
    }


class RenderFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.artifact = _synthetic_artifact()
        self.artifact_path = self.root / "artifact.json"
        self.artifact_path.write_text(json.dumps(self.artifact))

        self.doc_path = self.root / "doc.md"

        self._orig_artifact = r.ARTIFACT
        self._orig_blocks = dict(r.BLOCKS)
        r.ARTIFACT = self.artifact_path
        self.addCleanup(self._restore)

    def _restore(self):
        r.ARTIFACT = self._orig_artifact
        r.BLOCKS = self._orig_blocks

    def _write_doc(self, table="", reasons="", findings=""):
        self.doc_path.write_text(
            "# doc\n\n"
            "<!-- profile-421-generated: measured-towers-table -->\n"
            f"{table}\n"
            "<!-- /profile-421-generated -->\n\n"
            "<!-- profile-421-generated: candidate-reasons -->\n"
            f"{reasons}\n"
            "<!-- /profile-421-generated -->\n\n"
            "<!-- profile-421-generated: findings -->\n"
            f"{findings}\n"
            "<!-- /profile-421-generated -->\n"
        )
        r.BLOCKS = {
            "measured-towers-table": (self.doc_path, r.render_measured_towers_table),
            "candidate-reasons": (self.doc_path, r.render_candidate_reasons),
            "findings": (self.doc_path, r.render_findings),
        }


class RenderContentTests(RenderFixture):
    def test_measured_towers_table_renders_one_row_per_leg_in_order(self):
        table = r.render_measured_towers_table(self.artifact)
        rows = table.splitlines()
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("| CLIP-text | A1 | f32 |"))
        self.assertTrue(rows[1].startswith("| CLIP-text | D2 (LoRA+LN eager) | f32 |"))

    def test_d_leg_label_uses_canonical_kernel_order_not_json_order(self):
        """`kernels_disabled` lists `layer_norm_fused` BEFORE
        `lora_linear_fused` in this fixture -- the rendered label must still
        read `LoRA+LN`, the doc's own canonical order, not `LN+LoRA`."""
        table = r.render_measured_towers_table(self.artifact)
        self.assertIn("D2 (LoRA+LN eager)", table)
        self.assertNotIn("LN+LoRA", table)

    def test_a_leg_label_is_bare_suffix(self):
        table = r.render_measured_towers_table(self.artifact)
        self.assertIn("| A1 | f32 |", table)

    def test_table_precision_matches_the_doc(self):
        table = r.render_measured_towers_table(self.artifact)
        self.assertIn("0.1004 | 0.0000 | 0.0608 | 0.0396 | 0.0 | 60.5", table)

    def test_candidate_reasons_renders_one_bullet_per_decision(self):
        reasons = r.render_candidate_reasons(self.artifact)
        self.assertEqual(
            reasons, '- **`C-ATTN-clip-text`** — UNRESOLVED: "a short reason"'
        )

    def test_findings_renders_one_bullet_per_finding(self):
        findings = r.render_findings(self.artifact)
        self.assertEqual(findings, '- **`a-finding`**: "a short finding text"')

    def test_long_candidate_reason_wraps_at_the_docs_own_width(self):
        artifact = _synthetic_artifact()
        artifact["candidate_decisions"][0]["reason"] = (
            "neither ACTIVATE (s_wall>=10% on any decision-grade leg) nor DECLINE "
            "(combined share <5% on every decision-grade leg) — clip-text-A1: "
            "s_wall+U_wall=0.1030, s_busy+U_busy=0.1703; clip-text-A2: "
            "s_wall+U_wall=0.0878, s_busy+U_busy=0.2045"
        )
        rendered = r.render_candidate_reasons(artifact)
        self.assertEqual(
            rendered,
            '- **`C-ATTN-clip-text`** — UNRESOLVED: "neither ACTIVATE (s_wall>=10% on any decision-grade\n'
            "  leg) nor DECLINE (combined share <5% on every decision-grade leg) — clip-text-A1:\n"
            "  s_wall+U_wall=0.1030, s_busy+U_busy=0.1703; clip-text-A2: s_wall+U_wall=0.0878,\n"
            '  s_busy+U_busy=0.2045"',
        )


class CheckModeTests(RenderFixture):
    def test_check_passes_when_committed_matches_the_render(self):
        self._write_doc(
            table=r.render_measured_towers_table(self.artifact),
            reasons=r.render_candidate_reasons(self.artifact),
            findings=r.render_findings(self.artifact),
        )
        self.assertEqual(r.main(["--check"]), 0)

    def test_check_reds_on_a_one_digit_edit(self):
        table = r.render_measured_towers_table(self.artifact)
        mutated_table = table.replace("0.1004", "0.1005", 1)
        self._write_doc(
            table=mutated_table,
            reasons=r.render_candidate_reasons(self.artifact),
            findings=r.render_findings(self.artifact),
        )
        self.assertEqual(r.main(["--check"]), 1)

    def test_check_reds_on_a_stale_candidate_reason(self):
        self._write_doc(
            table=r.render_measured_towers_table(self.artifact),
            reasons='- **`C-ATTN-clip-text`** — UNRESOLVED: "a STALE reason"',
            findings=r.render_findings(self.artifact),
        )
        self.assertEqual(r.main(["--check"]), 1)

    def test_check_reds_on_a_stale_finding(self):
        self._write_doc(
            table=r.render_measured_towers_table(self.artifact),
            reasons=r.render_candidate_reasons(self.artifact),
            findings='- **`a-finding`**: "a STALE finding text"',
        )
        self.assertEqual(r.main(["--check"]), 1)

    def test_missing_marker_fails_closed(self):
        self.doc_path.write_text("# doc with no markers at all\n")
        r.BLOCKS = {"measured-towers-table": (self.doc_path, r.render_measured_towers_table)}
        self.assertEqual(r.main(["--check"]), 1)

    def test_default_mode_writes_the_render_into_the_doc(self):
        self._write_doc(table="STALE", reasons="STALE", findings="STALE")
        self.assertEqual(r.main([]), 0)
        text = self.doc_path.read_text()
        current = r._current_block(text, "measured-towers-table", self.doc_path)
        self.assertEqual(current, r.render_measured_towers_table(self.artifact))

    def test_write_then_check_is_idempotent(self):
        self._write_doc(table="STALE", reasons="STALE", findings="STALE")
        self.assertEqual(r.main([]), 0)
        self.assertEqual(r.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
