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
                "contract_valid": True,
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
                "contract_valid": True,
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
        "contract_validity": {
            "gate": "fixture",
            "legs_total": 2,
            "legs_valid": 2,
            "legs_failing": [],
        },
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

    def test_gate_column_marks_a_contract_valid_leg_ok(self):
        table = r.render_measured_towers_table(self.artifact)
        rows = table.splitlines()
        self.assertTrue(rows[0].endswith("| OK |"))
        self.assertTrue(rows[1].endswith("| OK |"))

    def test_gate_column_marks_a_failing_leg_from_contract_validity_never_hard_coded(self):
        """The `gate` column is read off `contract_validity.legs_failing`
        (cross-checked against `legs[].contract_valid` by
        `_contract_validity`), never a hard-coded leg id -- a leg the
        artifact itself marks failing renders `FAILS`, any renamed/future
        leg id included."""
        artifact = _synthetic_artifact()
        artifact["legs"][1]["contract_valid"] = False
        artifact["contract_validity"]["legs_valid"] = 1
        artifact["contract_validity"]["legs_failing"] = [
            {"leg_id": "clip-text-D2", "decision_grade_reason": "a fixture reason"}
        ]
        table = r.render_measured_towers_table(artifact)
        rows = table.splitlines()
        self.assertTrue(rows[0].endswith("| OK |"))
        self.assertTrue(rows[1].endswith("| FAILS |"))

    def test_gate_column_refuses_a_self_contradicting_artifact(self):
        """`render_measured_towers_table` must never render a plausible-
        looking `gate` column off an artifact whose own `contract_validity`
        block disagrees with `legs[].contract_valid` -- `_contract_validity`
        catches this before any row is built."""
        artifact = _synthetic_artifact()
        artifact["contract_validity"]["legs_valid"] = 2
        artifact["contract_validity"]["legs_failing"] = []
        artifact["legs"][1]["contract_valid"] = False
        with self.assertRaises(ValueError) as ctx:
            r.render_measured_towers_table(artifact)
        self.assertIn("disagrees", str(ctx.exception))

    def test_d_leg_unknown_kernels_disabled_key_is_refused_by_name(self):
        """`_leg_label` must refuse (never silently drop) a `kernels_
        disabled` key `_KERNEL_LABEL_ORDER` does not know -- the same
        refuse-by-name posture `_TOWER_DISPLAY`'s own dict lookup already
        has for an unrecognized tower."""
        artifact = _synthetic_artifact()
        artifact["legs"][1]["kernels_disabled"] = ["layer_norm_fused", "some_future_kernel_fused"]
        with self.assertRaises(ValueError) as ctx:
            r.render_measured_towers_table(artifact)
        self.assertIn("some_future_kernel_fused", str(ctx.exception))

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


def _full_synthetic_artifact() -> dict:
    """A second, richer fixture -- covers `realized_gains[]`/`attribution[]`,
    the two fields `_synthetic_artifact()` above never carries -- used only
    by the tests exercising the newer, non-table/non-verbatim-string
    blocks (realized gains, decision-grade note, DECLINE-band summary,
    the `htsat-A2` deviation, and their guide/CHANGELOG restatements).
    Deliberately round, distinguishable numbers (never the real repo's own
    measured values) so a test asserting against them can never pass by
    accident against the wrong source.
    """

    def chains(gelu_wall, gelu_busy, attn_busy, unattr_wall, unattr_busy, attn_tower):
        return {
            "C-GELU": {"share_wall": gelu_wall, "share_gpu_busy": gelu_busy},
            f"C-ATTN-{attn_tower}": {"share_gpu_busy": attn_busy},
            "UNATTRIBUTED": {"share_wall": unattr_wall, "share_gpu_busy": unattr_busy},
        }

    return {
        "limits": {"unattributed_decision_grade_limit": 0.05, "unknown_kernel_share_limit": 0.01},
        "legs": [
            {
                "leg_id": f"{t}-{a}",
                "dtype": "bf16" if a == "A2" else "f32",
                "verdict": "VALID",
                "merge_verdict": "VALID",
                "contract_valid": not (t == "htsat" and a == "A2"),
            }
            for t in ("clip-text", "clip-vision", "htsat")
            for a in ("A1", "A2", "D1", "D2")
        ],
        "contract_validity": {
            "gate": "fixture gate",
            "legs_total": 12,
            "legs_valid": 11,
            "legs_failing": [
                {
                    "leg_id": "htsat-A2",
                    "problems": ["htsat-A2: attribution decision_grade is False, not True"],
                    "decision_grade_reason": "UNATTRIBUTED share_gpu_busy=0.0566 > 0.05",
                }
            ],
        },
        "realized_gains": [
            {"chain": "C-LORA", "tower": "clip-text", "wall_delta_s_per_step": 0.010, "share_of_baseline_wall": 0.10},
            {"chain": "C-LORA", "tower": "clip-vision", "wall_delta_s_per_step": 0.020, "share_of_baseline_wall": 0.20},
            {"chain": "C-LORA", "tower": "htsat", "wall_delta_s_per_step": 0.030, "share_of_baseline_wall": 0.05},
            {"chain": "C-LN", "tower": "clip-text", "wall_delta_s_per_step": 0.005, "share_of_baseline_wall": 0.05},
            {"chain": "C-LN", "tower": "clip-vision", "wall_delta_s_per_step": 0.006, "share_of_baseline_wall": 0.06},
            {"chain": "C-LN", "tower": "htsat", "wall_delta_s_per_step": 0.007, "share_of_baseline_wall": 0.07, "note": "joint"},
        ],
        "attribution": [
            {"leg_id": "clip-text-A1", "verdict": "VALID", "decision_grade": True, "chains": chains(0.03, 0.04, 0.10, 0.01, 0.02, "clip-text")},
            {"leg_id": "clip-text-A2", "verdict": "VALID", "decision_grade": True, "chains": chains(0.031, 0.041, 0.11, 0.011, 0.021, "clip-text")},
            {"leg_id": "clip-vision-A1", "verdict": "VALID", "decision_grade": True, "chains": chains(0.032, 0.042, 0.12, 0.012, 0.022, "clip-vision")},
            {"leg_id": "clip-vision-A2", "verdict": "VALID", "decision_grade": True, "chains": chains(0.033, 0.043, 0.13, 0.013, 0.023, "clip-vision")},
            {
                "leg_id": "htsat-A2",
                "verdict": "VALID",
                "decision_grade": False,
                "decision_grade_reason": "UNATTRIBUTED share_gpu_busy=0.0566 > 0.05",
                "chains": {"UNATTRIBUTED": {"share_wall": 0.0, "share_gpu_busy": 0.0566}},
            },
            {
                "leg_id": "htsat-A1",
                "verdict": "VALID",
                "decision_grade": True,
                "decision_grade_reason": None,
                "chains": {"UNATTRIBUTED": {"share_wall": 0.0, "share_gpu_busy": 0.02}},
            },
            *[
                {"leg_id": f"{t}-{a}", "verdict": "VALID", "decision_grade": True, "chains": {}}
                for t, a in (
                    ("clip-text", "D1"), ("clip-text", "D2"),
                    ("clip-vision", "D1"), ("clip-vision", "D2"),
                    ("htsat", "D1"), ("htsat", "D2"),
                )
            ],
        ],
        "candidate_decisions": [
            {"port": "C-ATTN-clip-text", "verdict": "UNRESOLVED", "reason": "neither ACTIVATE nor DECLINE — clip-text-A1: s_wall+U_wall=0.11, s_busy+U_busy=0.12"},
            {"port": "C-MLP-clip-text", "verdict": "UNRESOLVED", "reason": "neither ACTIVATE nor DECLINE — clip-text-A1: s_wall+U_wall=0.04, s_busy+U_busy=0.06"},
            {"port": "C-ATTN-clip-vision", "verdict": "UNRESOLVED", "reason": "neither ACTIVATE nor DECLINE — clip-vision-A1: s_wall+U_wall=0.13, s_busy+U_busy=0.14"},
            {"port": "C-MLP-clip-vision", "verdict": "UNRESOLVED", "reason": "neither ACTIVATE nor DECLINE — clip-vision-A1: s_wall+U_wall=0.05, s_busy+U_busy=0.06"},
        ],
        "findings": [
            {
                "id": "htsat-front-end-bound",
                "text": "The HTSAT step: front-end share of wall is 80% on the F32 decision leg (htsat-A1).",
                "evidence": {
                    "front_share_of_wall_pct": {"htsat-A1": 80.0},
                    "front_end_bound": True,
                    "legs_read": ["htsat-A1"],
                    "legs_excluded": {"htsat-A2": "htsat-A2: attribution decision_grade is False, not True"},
                    "arm_invariant": True,
                    "invariance_axes": ["arm"],
                },
            },
            {
                "id": "clip-vision-front-end-share",
                "text": "CLIP-vision's own front end is 20-25% of wall on legs.",
                "evidence": {
                    "front_share_of_wall_pct": {"clip-vision-A1": 20.0, "clip-vision-A2": 25.0},
                },
            },
            {
                "id": "clip-launch-bound-batch8",
                "text": "3000-4000 launches/step; cuts GPU busy by 30-40%; wall drops by only 4-6%.",
                "evidence": {
                    "launches_per_step": {
                        "clip-text-A1": 3000.0, "clip-text-A2": 3100.0,
                        "clip-vision-A1": 3900.0, "clip-vision-A2": 4000.0,
                    },
                    "busy_delta_pct_bf16_vs_f32": {"clip-text": -30.0, "clip-vision": -40.0},
                    "wall_delta_pct_bf16_vs_f32": {"clip-text": -4.0, "clip-vision": -6.0},
                    "launch_bound": True,
                    "wall_drop_smaller_than_busy_drop_every_tower": True,
                },
            },
            {
                "id": "c-attn-htsat-out-of-tier",
                "text": "C-ATTN-HTSAT: 33% of GPU busy (~5% of wall).",
                "evidence": {"share_gpu_busy": 0.33, "share_wall": 0.05},
            },
        ],
        "suppressed_findings": [],
    }


def _norm(text: str) -> str:
    """Whitespace-collapsed `text` -- these blocks are `textwrap.fill`-ed at
    a fixed column width, so a substring assertion must not care which
    exact line a word-wrap boundary landed the space on."""
    return " ".join(text.split())


class NewBlockRenderTests(unittest.TestCase):
    def setUp(self):
        self.artifact = _full_synthetic_artifact()

    def test_realized_gains_reads_all_three_bullets_from_the_artifact(self):
        rendered = _norm(r.render_realized_gains(self.artifact))
        self.assertIn("+10.0 ms/step (10.0 % of the A1 baseline wall)", rendered)
        self.assertIn("+20.0 ms/step (20.0 %)", rendered)
        self.assertIn("HTSAT +30.0 ms/step (5.0 %)", rendered)
        self.assertIn("+5.0 ms/step (5.0 % of A1 baseline wall)", rendered)
        self.assertIn("+6.0 ms/step (6.0 %)", rendered)
        self.assertIn("C-LN + C-GELU-HTSAT joint", rendered)
        self.assertIn("+7.0 ms/step (7.0 % of the A1 baseline wall)", rendered)

    def test_realized_gains_guide_and_readme_agree_on_the_same_numbers(self):
        readme_block = _norm(r.render_realized_gains(self.artifact))
        guide_block = _norm(r.render_realized_gains_guide(self.artifact))
        for token in ("+10.0 ms", "+20.0 ms", "+30.0 ms", "+5.0 ms", "+6.0 ms", "+7.0 ms"):
            self.assertIn(token, readme_block)
            self.assertIn(token, guide_block)

    def test_decision_grade_note_renders_when_expectations_hold(self):
        rendered = r.render_decision_grade_note(self.artifact)
        self.assertIn("UNRESOLVED", rendered)
        self.assertIn("htsat-A2", rendered)

    def test_decision_grade_note_refuses_when_a_clip_a2_leg_is_not_decision_grade(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][1]["decision_grade"] = False  # clip-text-A2
        with self.assertRaises(ValueError) as ctx:
            r.render_decision_grade_note(artifact)
        self.assertIn("decision-grade", str(ctx.exception))

    def test_decision_grade_note_refuses_when_htsat_a2_becomes_decision_grade(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][4]["decision_grade"] = True  # htsat-A2
        with self.assertRaises(ValueError):
            r.render_decision_grade_note(artifact)

    def test_decline_band_summary_computes_combined_shares_never_hand_adds(self):
        rendered = _norm(r.render_decline_band_summary(self.artifact))
        # gelu_wall_pct('clip-text', 'A1') == 3.00 %, combined wall == (0.03+0.01)*100 == 4.00 %
        self.assertIn("3.00 %/3.10 %", rendered)
        self.assertIn("4.00 %/4.20 %", rendered)

    def test_htsat_a2_deviation_reads_the_unattributed_share(self):
        rendered = r.render_htsat_a2_deviation(self.artifact)
        self.assertIn("5.66 %", rendered)

    def test_changelog_entry_refuses_when_a_leg_is_not_valid(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][0]["verdict"] = "INVALID"
        with self.assertRaises(ValueError):
            r.render_changelog_421_entry(artifact)

    def test_findings_guide_extracts_every_percentage_from_the_findings_text(self):
        rendered = _norm(r.render_findings_guide(self.artifact))
        self.assertIn("80 %", rendered)
        self.assertIn("on the F32 decision leg (`htsat-A1`; `htsat-A2` excluded: fails the contract's validity gate)", _norm(rendered))
        self.assertIn("arm-invariant", rendered)
        self.assertNotIn("dtype- and arm-invariant", rendered)
        self.assertIn("20–25 %", rendered)
        self.assertIn("30–40 %", rendered)
        self.assertIn("4–6 %", rendered)
        self.assertIn("33 %", rendered)

    def test_findings_guide_refuses_when_the_finding_it_needs_is_absent(self):
        """A finding this renderer's fixed prose depends on going missing
        entirely (e.g. suppressed) must fail closed, never render around
        the gap."""
        artifact = _full_synthetic_artifact()
        artifact["findings"] = [f for f in artifact["findings"] if f["id"] != "htsat-front-end-bound"]
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("htsat-front-end-bound", str(ctx.exception))

    def _withhold(self, finding_id: str, flag: str):
        artifact = _full_synthetic_artifact()
        for finding in artifact["findings"]:
            if finding["id"] == finding_id:
                finding["evidence"][flag] = False
        return artifact

    def _absent(self, finding_id: str, flag: str):
        artifact = _full_synthetic_artifact()
        for finding in artifact["findings"]:
            if finding["id"] == finding_id:
                del finding["evidence"][flag]
        return artifact

    # -- B3: the guide and CHANGELOG renderers must take every licensed
    # word from the artifact's own `evidence`, and fail CLOSED (never
    # render the word anyway) the moment a run's own evidence withholds it
    # -- one test per word per surface (`findings-guide`, CHANGELOG).

    def test_findings_guide_refuses_when_front_end_bound_is_withheld(self):
        artifact = self._withhold("htsat-front-end-bound", "front_end_bound")
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("front_end_bound", str(ctx.exception))
        self.assertIn("CPU front-end-bound", str(ctx.exception))

    def test_findings_guide_refuses_when_arm_invariance_is_absent(self):
        """Absent (never evaluated -- e.g. the D-arm legs were not all
        VALID) must refuse exactly like an explicit `False`, never be
        treated as 'not applicable, render the word anyway'."""
        artifact = self._absent("htsat-front-end-bound", "arm_invariant")
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("arm_invariant", str(ctx.exception))
        self.assertIn("invariant", str(ctx.exception))

    def test_findings_guide_refuses_when_launch_bound_is_withheld(self):
        artifact = self._withhold("clip-launch-bound-batch8", "launch_bound")
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("launch_bound", str(ctx.exception))
        self.assertIn("launch-bound", str(ctx.exception))

    def test_magnitude_range_refuses_on_a_non_negative_value(self):
        """`_magnitude_range` gates the 'cuts'/'drops' direction verbs on an
        explicit, live sign check of the mapping it is handed -- a
        non-negative value (BF16 did NOT decrease this tower's number) must
        fail closed, never silently render a magnitude with the wrong
        implied direction."""
        with self.assertRaises(ValueError) as ctx:
            r._magnitude_range({"clip-text": -30.0, "clip-vision": 5.0})
        self.assertIn("not uniformly negative", str(ctx.exception))

    def test_magnitude_range_accepts_uniformly_negative_values(self):
        self.assertEqual(r._magnitude_range({"clip-text": -30.0, "clip-vision": -40.0}), "30–40")

    def _sign_flip_launch_delta(self, key: str):
        """A copy of `_full_synthetic_artifact()` where one tower's own
        `clip-launch-bound-batch8` delta (`busy_delta_pct_bf16_vs_f32` or
        `wall_delta_pct_bf16_vs_f32`) has flipped sign (BF16 GREW that
        tower's number instead of shrinking it) -- the shape a run whose
        own signed deltas do not license the "cuts"/"drops" blanket verb
        would actually produce."""
        artifact = _full_synthetic_artifact()
        for finding in artifact["findings"]:
            if finding["id"] == "clip-launch-bound-batch8":
                finding["evidence"][key]["clip-vision"] = abs(finding["evidence"][key]["clip-vision"])
        return artifact

    def test_findings_guide_refuses_when_a_busy_delta_is_not_negative(self):
        artifact = self._sign_flip_launch_delta("busy_delta_pct_bf16_vs_f32")
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("not uniformly negative", str(ctx.exception))

    def test_findings_guide_refuses_when_a_wall_delta_is_not_negative(self):
        artifact = self._sign_flip_launch_delta("wall_delta_pct_bf16_vs_f32")
        with self.assertRaises(ValueError) as ctx:
            r.render_findings_guide(artifact)
        self.assertIn("not uniformly negative", str(ctx.exception))

    def test_changelog_refuses_when_a_busy_delta_is_not_negative(self):
        artifact = self._sign_flip_launch_delta("busy_delta_pct_bf16_vs_f32")
        for leg in artifact["attribution"]:
            leg["verdict"] = "VALID"
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("not uniformly negative", str(ctx.exception))

    def test_changelog_refuses_when_front_end_bound_is_withheld(self):
        artifact = self._withhold("htsat-front-end-bound", "front_end_bound")
        for leg in artifact["attribution"]:
            leg["verdict"] = "VALID"
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("front_end_bound", str(ctx.exception))

    def test_changelog_refuses_when_arm_invariance_is_absent(self):
        artifact = self._absent("htsat-front-end-bound", "arm_invariant")
        for leg in artifact["attribution"]:
            leg["verdict"] = "VALID"
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("arm_invariant", str(ctx.exception))

    def test_changelog_refuses_when_launch_bound_is_withheld(self):
        artifact = self._withhold("clip-launch-bound-batch8", "launch_bound")
        for leg in artifact["attribution"]:
            leg["verdict"] = "VALID"
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("launch_bound", str(ctx.exception))

    # -- every fixed verdict/relational word is licensed by the artifact's own numbers --

    def _activate(self, index: int = 0):
        artifact = _full_synthetic_artifact()
        artifact["candidate_decisions"][index]["verdict"] = "ACTIVATE"
        return artifact

    def test_every_unresolved_renderer_refuses_when_a_candidate_activates(self):
        for render in (
            r.render_decision_grade_note,
            r.render_decline_band_summary,
            r.render_decline_band_guide,
            r.render_changelog_421_entry,
        ):
            with self.subTest(renderer=render.__name__):
                with self.assertRaises(ValueError) as ctx:
                    render(self._activate())
                self.assertIn("not UNRESOLVED", str(ctx.exception))
                with self.assertRaises(ValueError):
                    render(self._activate(3))

    def test_decline_band_renderers_refuse_when_a_wall_share_is_not_under_the_floor(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][0]["chains"]["C-GELU"]["share_wall"] = 0.061  # clip-text-A1: 6.1 % >= 5 %
        for render in (r.render_decline_band_summary, r.render_decline_band_guide, r.render_changelog_421_entry):
            with self.subTest(renderer=render.__name__):
                with self.assertRaises(ValueError) as ctx:
                    render(artifact)
                self.assertIn("not below 5 %", str(ctx.exception))

    def test_decline_band_renderers_refuse_when_a_busy_share_leaves_the_band(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][1]["chains"]["C-GELU"]["share_gpu_busy"] = 0.20  # clip-text-A2: 22.1 % combined
        for render in (r.render_decline_band_summary, r.render_decline_band_guide, r.render_changelog_421_entry):
            with self.subTest(renderer=render.__name__):
                with self.assertRaises(ValueError) as ctx:
                    render(artifact)
                self.assertIn("not in the 5-10 % band", str(ctx.exception))

    def test_changelog_refuses_when_an_attention_busy_share_falls_inside_the_band(self):
        artifact = _full_synthetic_artifact()
        artifact["attribution"][0]["chains"]["C-ATTN-clip-text"]["share_gpu_busy"] = 0.05  # 7 % combined
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("inside the 5-10 % band", str(ctx.exception))

    def test_changelog_headline_counts_the_contract_gate_not_the_merge_verdict(self):
        rendered = _norm(r.render_changelog_421_entry(self.artifact))
        self.assertIn(
            "11 of 12 legs (`A1`/`A2`/`D1`/`D2` × CLIP-text, OpenCLIP-vision, HTSAT) pass the contract's "
            "validity gate — `htsat-A2` fails it (UNATTRIBUTED share_gpu_busy=0.0566 > 0.05) and is excluded "
            "from every finding on",
            rendered,
        )
        self.assertNotIn("are VALID", rendered)
        self.assertIn("80 % of wall on the F32 decision leg", rendered)
        self.assertIn("arm-invariant)", rendered)
        self.assertNotIn("dtype- and arm-invariant", rendered)

    def test_contract_validity_headline_refuses_an_artifact_that_contradicts_itself(self):
        artifact = _full_synthetic_artifact()
        artifact["contract_validity"]["legs_valid"] = 12  # says all pass, legs[] says htsat-A2 does not
        with self.assertRaises(ValueError):
            r.render_changelog_421_entry(artifact)
        artifact = _full_synthetic_artifact()
        del artifact["contract_validity"]
        with self.assertRaises(ValueError) as ctx:
            r.render_changelog_421_entry(artifact)
        self.assertIn("contract_validity", str(ctx.exception))

    def test_measured_summary_names_the_failing_leg_and_all_pass_when_none_fails(self):
        base = {
            "status": "GREEN",
            "git_sha": "f" * 40,
            "p2_witnessed": {"towers": ["clip-text", "clip-vision", "htsat"]},
            "notes": {"gpu": "FIXTURE-GPU", "driver": "0.0.0", "nsys": "Nsight Systems 1.2.3 (fixture)"},
        }
        artifact = _full_synthetic_artifact()
        for leg in artifact["legs"]:
            leg["tower"] = leg["leg_id"].rsplit("-", 1)[0]
        artifact.update(base)
        rendered = _norm(r.render_measured_summary(artifact))
        self.assertIn(
            "11 of 12 legs pass the contract's validity gate — `htsat-A2` fails it (UNATTRIBUTED "
            "share_gpu_busy=0.0566 > 0.05) and is excluded from every finding; the BF16 pre-flight",
            rendered,
        )
        for leg in artifact["legs"]:
            leg["contract_valid"] = True
        artifact["contract_validity"] = {"gate": "fixture", "legs_total": 12, "legs_valid": 12, "legs_failing": []}
        rendered = _norm(r.render_measured_summary(artifact))
        self.assertIn("All 12 legs pass the contract's validity gate; the BF16 pre-flight", rendered)

    def test_htsat_a2_deviation_refuses_when_the_finding_still_reads_htsat_a2(self):
        artifact = _full_synthetic_artifact()
        for finding in artifact["findings"]:
            if finding["id"] == "htsat-front-end-bound":
                finding["evidence"]["legs_excluded"] = {}
                finding["evidence"]["legs_read"] = ["htsat-A1", "htsat-A2"]
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("excluded", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_the_finding_claims_dtype_invariance(self):
        artifact = _full_synthetic_artifact()
        for finding in artifact["findings"]:
            if finding["id"] == "htsat-front-end-bound":
                finding["evidence"]["invariance_axes"] = ["dtype", "arm"]
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("dtype-invariance", str(ctx.exception))

    def test_htsat_a2_deviation_names_the_consequence(self):
        rendered = _norm(r.render_htsat_a2_deviation(self.artifact))
        self.assertIn("is merge-VALID but fails the contract's validity gate (not decision-grade for attribution)", rendered)
        self.assertIn("the HTSAT front-end finding reads `htsat-A1` only and asserts no dtype-invariance.", rendered)

    def test_htsat_a2_deviation_refuses_when_decision_grade_becomes_true(self):
        """The bullet's own fixed prose ('is VALID but not decision-grade')
        is licensed by `attribution[htsat-A2].decision_grade` itself, never
        a hard-coded negation -- a run where this leg becomes decision-grade
        must fail this renderer closed, never keep asserting the opposite."""
        artifact = _full_synthetic_artifact()
        artifact["attribution"][4]["decision_grade"] = True  # htsat-A2
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("decision_grade", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_merge_verdict_is_not_valid(self):
        artifact = _full_synthetic_artifact()
        for leg in artifact["legs"]:
            if leg["leg_id"] == "htsat-A2":
                leg["merge_verdict"] = "INVALID"
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("merge_verdict", str(ctx.exception))

    def test_htsat_a2_deviation_reads_the_bound_off_the_artifacts_limits_block(self):
        """The "N %" bound is read off the artifact's OWN top-level
        `limits.unattributed_decision_grade_limit` (never a live import of
        the module constant, and never re-parsed out of `attribution[htsat-
        A2].decision_grade_reason` -- that string is still cross-checked
        for a SECOND, independent copy of `share`, but no longer for the
        bound itself); the comparison word is a live `>` comparison against
        that SAME artifact-sourced bound -- this fixture's `limits` (5 %)
        and share (5.66 %) must still render "over the ... 5 % validity
        bound"."""
        rendered = _norm(r.render_htsat_a2_deviation(self.artifact))
        self.assertIn("over the contract's 5 % validity bound", rendered)

    def test_htsat_a2_deviation_refuses_when_limits_block_is_absent(self):
        artifact = _full_synthetic_artifact()
        del artifact["limits"]
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("limits", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_limits_disagrees_with_the_reason_share(self):
        """The artifact's own `limits` bound (5 %) and its own
        `decision_grade_reason` share (5.66 %, over 5 %) must independently
        agree on DIRECTION -- an artifact whose `limits` bound was edited to
        exceed the recorded share (a self-contradictory input this renderer
        never received in practice, since `profile_421_artifact.py` itself
        cross-checks `limits` against its own live constant) must still
        fail this renderer CLOSED rather than silently flip the comparison
        word."""
        artifact = _full_synthetic_artifact()
        artifact["limits"]["unattributed_decision_grade_limit"] = 0.90  # now over htsat-A2's 5.66 % share
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("under its own bound", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_decision_grade_reason_is_not_the_expected_shape(self):
        artifact = _full_synthetic_artifact()
        for leg in artifact["attribution"]:
            if leg["leg_id"] == "htsat-A2":
                leg["decision_grade_reason"] = "UNATTRIBUTED share_gpu_busy is non-finite (nan)"
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("decision_grade_reason", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_reason_share_disagrees_with_chains_share(self):
        artifact = _full_synthetic_artifact()
        for leg in artifact["attribution"]:
            if leg["leg_id"] == "htsat-A2":
                leg["decision_grade_reason"] = "UNATTRIBUTED share_gpu_busy=0.9000 > 0.05"
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("disagrees", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_the_share_is_not_actually_over_the_bound(self):
        """A `decision_grade_reason` claiming a "<" or "=" relationship (a
        different failing rule than the "over the bound" one this bullet's
        fixed prose asserts) must fail closed, never keep asserting "over"."""
        artifact = _full_synthetic_artifact()
        for leg in artifact["attribution"]:
            if leg["leg_id"] == "htsat-A2":
                leg["decision_grade_reason"] = "UNATTRIBUTED share_gpu_busy=0.0400 > 0.05"
                leg["chains"]["UNATTRIBUTED"]["share_gpu_busy"] = 0.0400
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("under its own bound", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_htsat_a1_does_not_actually_clear_the_bound(self):
        artifact = _full_synthetic_artifact()
        for leg in artifact["attribution"]:
            if leg["leg_id"] == "htsat-A1":
                leg["chains"]["UNATTRIBUTED"]["share_gpu_busy"] = 0.9
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("does not actually clear", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_htsat_a1_decision_grade_is_false(self):
        artifact = _full_synthetic_artifact()
        for leg in artifact["attribution"]:
            if leg["leg_id"] == "htsat-A1":
                leg["decision_grade"] = False
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("htsat-A1", str(ctx.exception))

    def test_htsat_a2_deviation_refuses_when_suppressed_findings_contradicts_htsat_a1(self):
        """`attribution[htsat-A1].decision_grade` says `True`, but this
        artifact's own `suppressed_findings` names `htsat-A1` as not
        decision-grade -- an artifact contradicting itself, never a case
        this bullet may render around."""
        artifact = _full_synthetic_artifact()
        artifact["suppressed_findings"] = [
            {
                "id": "some-other-finding",
                "legs": ["htsat-A1"],
                "reason": "htsat-A1: attribution decision_grade is False, not True",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("contradict", str(ctx.exception))


class RealArtifactHtsatA2DeviationTests(unittest.TestCase):
    """`render_htsat_a2_deviation` exercised directly against the REAL
    committed close-out artifact and the REAL committed README bullet
    (never the synthetic fixture) -- the refactor that made the comparison
    word and the bound artifact-derived must not move this bullet's own
    committed text by one byte, and a deliberately contradictory copy of
    the SAME real artifact must still fail this renderer closed.
    """

    def setUp(self):
        self.artifact = json.loads(r.ARTIFACT.read_text())

    def test_matches_the_committed_real_block_byte_for_byte(self):
        committed = r._current_block(r.README.read_text(), "htsat-a2-deviation", r.README)
        rendered = r.render_htsat_a2_deviation(self.artifact)
        self.assertEqual(rendered, committed)

    def test_refuses_on_a_contradictory_copy_of_the_real_artifact(self):
        artifact = json.loads(json.dumps(self.artifact))  # deep copy
        artifact["suppressed_findings"] = list(artifact.get("suppressed_findings", [])) + [
            {
                "id": "some-other-finding",
                "legs": ["htsat-A1"],
                "reason": "htsat-A1: attribution decision_grade is False, not True",
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            r.render_htsat_a2_deviation(artifact)
        self.assertIn("contradict", str(ctx.exception))


class LiveSourceTests(unittest.TestCase):
    """`corpus_pool_counts`/`hermetic_test_count`/`basename_ambiguity_counts`
    read this REAL repo's own live source files -- never the synthetic
    artifact fixture -- so these are cheap real-repo smoke checks, the same
    shape `test_check_citations.py`'s own `test_real_contract_epoch_and_
    scope_are_recognized` already is.
    """

    def test_corpus_pool_counts_match_the_live_driver_and_producer_constants(self):
        total_files, train_clips, m_leg_rows = r.corpus_pool_counts()
        self.assertEqual((total_files, train_clips, m_leg_rows), (24, 16, 4800))

    def test_hermetic_test_count_matches_an_independent_discovery_pass(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "independent_test_profile_421_merge", r.PROFILE_421_MERGE_TEST
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = unittest.TestLoader().loadTestsFromModule(module).countTestCases()
        self.assertEqual(r.hermetic_test_count(), expected)
        self.assertGreater(expected, 0)

    def test_basename_ambiguity_counts_match_the_pinned_epoch_tree(self):
        epoch_sha, layer_norm_count, main_count = r.basename_ambiguity_counts()
        self.assertTrue(epoch_sha.startswith("bff1fad6"))
        self.assertEqual((layer_norm_count, main_count), (3, 10))


class RealRepoCheckTests(unittest.TestCase):
    """The master regression check: every block this script owns, rendered
    from the REAL committed artifact, must match the REAL committed text in
    README/the guide/CHANGELOG.md -- exercised by every other test above
    only through a synthetic fixture, this is the one test that proves the
    wiring in the actual tracked docs is live and green.
    """

    def test_check_passes_against_the_real_repo_docs(self):
        self.assertEqual(r.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
