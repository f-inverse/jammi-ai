#!/usr/bin/env python3
"""Hermetic tests for `profile_421_attribute.py` (issue #421 post-export
attribution unit; CONTRACT `scratchpad/contract-421-profile.md` v2.5
`### Attribution` / `§D3`).

No GPU, no pod, no `nsys`, no network. The kernel-name<->shape MAPPING
itself is tested against a SMALL, REAL, committed fixture
(`fixtures/profile_421_clip_text_a1/kernels.json`, cut byte-for-byte from
the real `clip-text-A1` leg's nsys census export — see that directory's
`PROVENANCE.md`), never a hand-rolled census standing in for what a real
export actually contains. Per this crate's "never transcribe a timing
number into a test" convention, NOTHING here asserts a literal
`us_per_step`/`share` value from that fixture as an expected number — every
assertion is STRUCTURAL: which chain a `(kernel, grid, block)` row lands in,
that chain shares partition `gpu_kernel_us_per_step` exactly, that shares
are `<= 1`, that every declared chain is present-or-explicitly-absent, and
that the badd/sgemm mapping decisions documented in the module doc hold.

Run: `python3 ci/scripts/perf/test_profile_421_attribute.py`
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
ATTRIBUTE = PERF_DIR / "profile_421_attribute.py"
FIXTURE_KERNELS = PERF_DIR / "fixtures" / "profile_421_clip_text_a1" / "kernels.json"

sys.path.insert(0, str(PERF_DIR))
import profile_421_attribute as attribute  # noqa: E402


# The REAL leg parameters `clip-text-A1`'s own `manifest.json` recorded
# (batch=8, max_seq_length=77, fusible_site_census.lora_sites_wrapped=48) —
# see `fixtures/profile_421_clip_text_a1/PROVENANCE.md`.
CLIP_TEXT_A1_MANIFEST_FIELDS = {
    "tower": "clip-text",
    "dtype": "f32",
    "batch": 8,
    "max_seq_length": 77,
    "kernels_disabled": [],
    "census_ok": True,
    "census_exit": 0,
    "fusible_site_census": {
        "n": {"lora_sites_wrapped": 48, "layer_norms": 25, "gelu_seam_calls_per_forward": 0},
        "m": {"lora_sites_wrapped": 48, "layer_norms": 25, "gelu_seam_calls_per_forward": 0},
    },
}


def load_fixture_census() -> dict:
    with FIXTURE_KERNELS.open(encoding="utf-8") as f:
        return json.load(f)


def clip_text_signatures() -> attribute.TowerSignatures:
    return attribute.derive_signatures("clip-text", CLIP_TEXT_A1_MANIFEST_FIELDS)


class DeriveSignaturesTests(unittest.TestCase):
    def test_clip_text_witnessed_and_declared_fields(self):
        sig = clip_text_signatures()
        self.assertEqual(sig.rows.value, 24)
        self.assertIn("witnessed", sig.rows.source)
        self.assertEqual(sig.seq.value, 77)
        self.assertIn("witnessed", sig.seq.source)
        self.assertEqual(sig.width.value, 512)
        self.assertIn("declared_constant", sig.width.source)
        self.assertEqual(sig.heads.value, 8)
        self.assertEqual(sig.mlp_width.value, 2048)
        self.assertEqual(sig.layers.value, 12)
        self.assertIn("witnessed", sig.layers.source)

    def test_clip_vision_uses_declared_seq_constant(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        manifest["tower"] = "clip-vision"
        manifest.pop("max_seq_length")
        sig = attribute.derive_signatures("clip-vision", manifest)
        self.assertEqual(sig.seq.value, 50)
        self.assertIn("declared_constant", sig.seq.source)
        self.assertEqual(sig.width.value, 768)
        self.assertEqual(sig.heads.value, 12)

    def test_derived_element_counts_match_the_real_export(self):
        # These three numbers are the ones `PROVENANCE.md` documents the
        # fixture rows were chosen to sit exactly at.
        sig = clip_text_signatures()
        self.assertEqual(sig.gelu_shape_elements(), 1848 * 2048)
        self.assertEqual(sig.attn_softmax_rows(), 24 * 8 * 77)
        self.assertEqual(sig.attn_batch_count(), 24 * 8)

    def test_unknown_tower_raises(self):
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("bert", CLIP_TEXT_A1_MANIFEST_FIELDS)

    def test_missing_batch_raises(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        del manifest["batch"]
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("clip-text", manifest)

    def test_non_multiple_lora_sites_raises(self):
        manifest = json.loads(json.dumps(CLIP_TEXT_A1_MANIFEST_FIELDS))
        manifest["fusible_site_census"]["n"]["lora_sites_wrapped"] = 49
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("clip-text", manifest)

    def test_flat_fusible_site_census_shape_also_works(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        manifest["fusible_site_census"] = {
            "lora_sites_wrapped": 48,
            "layer_norms": 25,
            "gelu_seam_calls_per_forward": 0,
        }
        sig = attribute.derive_signatures("clip-text", manifest)
        self.assertEqual(sig.layers.value, 12)


class ClassifyKernelUnitTests(unittest.TestCase):
    """Per-row classification, isolated from the whole-census pass — one
    assertion per mapping decision documented in the module doc."""

    def setUp(self):
        self.sig = clip_text_signatures()

    def test_layer_norm_kernels_match_by_name_at_any_grid(self):
        for name in ("layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"):
            entry = {"kernel": name, "grid": [1, 1, 1], "block": [1, 1, 1]}
            self.assertEqual(
                attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_LN
            )

    def test_usigmoid_matches_gelu_at_any_grid(self):
        entry = {"kernel": "usigmoid_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_GELU
        )

    def test_affine_and_bmul_match_gelu_only_at_declared_shape(self):
        elements = self.sig.gelu_shape_elements()
        at_shape = {"kernel": "affine_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(elements, 3696 * 1024)
        self.assertEqual(
            attribute.classify_kernel(at_shape, self.sig, "clip-text"), attribute.CHAIN_GELU
        )
        wrong_shape = {"kernel": "bmul_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(wrong_shape, self.sig, "clip-text"))

    def test_badd_at_the_gelu_shape_is_not_gelu(self):
        """Negative control for the documented exclusion: a bias-add at the
        identical element count as quick_gelu's own tensor is the Linear
        layer's bias, not the activation."""
        entry = {"kernel": "badd_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_attn_reduction_kernels_match_only_at_declared_row_count(self):
        rows = self.sig.attn_softmax_rows()
        self.assertEqual(rows, 14784)
        at_rows = {"kernel": "fast_max_f32", "grid": [14784, 1, 1], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(at_rows, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )
        other_rows = {"kernel": "fast_sum_f32", "grid": [1, 1, 1], "block": [1024, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(other_rows, self.sig, "clip-text"))

    def test_attn_reduction_requires_a_1d_grid(self):
        entry = {"kernel": "fast_max_f32", "grid": [14784, 2, 1], "block": [128, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_ampere_sgemm_matches_attn_when_grid_carries_rows_times_heads(self):
        batch = self.sig.attn_batch_count()
        self.assertEqual(batch, 192)
        entry = {"kernel": "ampere_sgemm_128x128_nt", "grid": [1, 1, 192], "block": [256, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_ampere_sgemm_without_the_batch_count_is_unattributed(self):
        """Negative control: a base/LoRA Linear projection's 2-D matmul,
        real row from the fixture, must NOT be classified as attention."""
        entry = {"kernel": "ampere_sgemm_128x64_nn", "grid": [4, 29, 7], "block": [128, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_unrelated_known_kernels_are_unattributed(self):
        for name, grid in (
            ("dropout_fwd_f32", [3696, 1, 1]),
            ("adamw_moment_update_f32", [4, 1, 1]),
        ):
            entry = {"kernel": name, "grid": grid, "block": [1024, 1, 1]}
            self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))


class AttributeCensusRealFixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed 15-row cut."""

    def setUp(self):
        self.census = load_fixture_census()
        self.sig = clip_text_signatures()

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(
            c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None
        )
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_every_declared_chain_is_present(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        for name in (attribute.CHAIN_LN, attribute.CHAIN_GELU, attribute.chain_attn("clip-text")):
            self.assertIn(name, chains)
            self.assertEqual(chains[name].status, "measured")
        self.assertIn(attribute.CHAIN_UNATTRIBUTED, chains)

    def test_shares_are_bounded_by_one(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        for name, result in chains.items():
            if result.share_gpu_busy is not None:
                self.assertGreaterEqual(result.share_gpu_busy, 0.0, name)
                self.assertLessEqual(result.share_gpu_busy, 1.0, name)
            if result.share_wall is not None:
                self.assertGreaterEqual(result.share_wall, 0.0, name)

    def test_ln_kernel_names_are_exactly_the_two_ln_kernels(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"},
        )

    def test_gelu_kernel_names_exclude_badd(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        gelu_names = set(chains[attribute.CHAIN_GELU].kernel_names)
        self.assertEqual(gelu_names, {"usigmoid_f32", "affine_f32", "bmul_f32"})
        self.assertNotIn("badd_f32", gelu_names)

    def test_attn_kernel_names(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        attn_names = set(chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertEqual(
            attn_names,
            {
                "fast_max_f32",
                "fast_sum_f32",
                "ampere_sgemm_128x128_nt",
                "ampere_sgemm_128x128_tn",
                "ampere_sgemm_128x128_nn",
            },
        )

    def test_unattributed_absorbs_the_negative_controls(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        unattributed_names = set(chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)
        self.assertIn("badd_f32", unattributed_names)
        self.assertIn("ampere_sgemm_128x64_nn", unattributed_names)
        self.assertIn("dropout_fwd_f32", unattributed_names)
        self.assertIn("adamw_moment_update_f32", unattributed_names)
        # The second `bmul_f32` row (wrong shape) contributes here too, but
        # `kernel_names` is a de-duplicated set shared with the row that DID
        # match GELU, so this is checked via the row-level unit test above
        # (`test_affine_and_bmul_match_gelu_only_at_declared_shape`) rather
        # than re-derived from the name set here.

    def test_empty_census_leaves_ln_gelu_attn_absent(self):
        empty = {
            "gpu_kernel_us_per_step": 0.0,
            "wall_s_per_step": 0.0,
            "by_kernel_and_grid": [],
        }
        chains, unknown, reasons = attribute.attribute_census(empty, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
        for name in (attribute.CHAIN_LN, attribute.CHAIN_GELU, attribute.chain_attn("clip-text")):
            self.assertEqual(chains[name].status, "absent")
            self.assertIsNone(chains[name].gpu_busy_us)
        self.assertEqual(chains[attribute.CHAIN_UNATTRIBUTED].status, "measured")
        self.assertEqual(chains[attribute.CHAIN_UNATTRIBUTED].gpu_busy_us, 0.0)


class UnknownKernelGateTests(unittest.TestCase):
    def setUp(self):
        self.sig = clip_text_signatures()

    def _census_with(self, extra_row: dict) -> dict:
        rows = list(load_fixture_census()["by_kernel_and_grid"])
        rows.append(extra_row)
        total = sum(r["us_per_step"] for r in rows)
        return {
            "gpu_kernel_us_per_step": total,
            "wall_s_per_step": 0.1,
            "by_kernel_and_grid": rows,
        }

    def test_unknown_kernel_above_threshold_invalidates(self):
        census = self._census_with(
            {"kernel": "totally_new_kernel_f32", "grid": [1, 1, 1], "block": [1, 1, 1], "us_per_step": 1e9}
        )
        chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertTrue(any("totally_new_kernel_f32" in r for r in reasons))
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "totally_new_kernel_f32")
        self.assertGreater(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        # An unknown kernel still contributes its time to UNATTRIBUTED —
        # the leg is flagged INVALID by the caller via `reasons`, but the
        # busy accounting itself stays a complete partition.
        self.assertIn("totally_new_kernel_f32", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_unknown_kernel_below_threshold_is_recorded_but_not_invalidating(self):
        census = self._census_with(
            {"kernel": "tiny_new_kernel_f32", "grid": [1, 1, 1], "block": [1, 1, 1], "us_per_step": 0.0001}
        )
        chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertEqual(reasons, [])
        self.assertEqual(len(unknown), 1)
        self.assertLess(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)


class LegRoleTests(unittest.TestCase):
    def test_a_leg_has_no_disabled_keys(self):
        self.assertEqual(attribute.leg_role([], "clip-text"), "A")

    def test_d2_leg_disables_only_lora_linear_fused(self):
        self.assertEqual(attribute.leg_role(["lora_linear_fused"], "clip-text"), "D2")

    def test_d1_leg_is_per_tower(self):
        self.assertEqual(
            attribute.leg_role(["lora_linear_fused", "layer_norm_fused"], "clip-text"), "D1"
        )
        self.assertEqual(
            attribute.leg_role(
                ["lora_linear_fused", "layer_norm_fused", "gelu_erf_fused"], "htsat"
            ),
            "D1",
        )
        # The CLIP D1 set named on an HTSAT leg is neither tower's D1/D2 set.
        self.assertEqual(
            attribute.leg_role(["lora_linear_fused", "layer_norm_fused"], "htsat"), "other"
        )

    def test_unrecognised_disable_set_is_other(self):
        self.assertEqual(attribute.leg_role(["some_other_key"], "clip-text"), "other")


def _chain_result(**kwargs) -> dict:
    return attribute.ChainResult(**kwargs).as_dict()


class LoraD2DeltaTests(unittest.TestCase):
    def _leg(self, leg_id, tower, role, verdict, busy, dtype="f32", wall=0.1):
        return {
            "leg_id": leg_id,
            "tower": tower,
            "dtype": dtype,
            "verdict": verdict,
            "reasons": [],
            "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
            "_role": role,
            "_gpu_busy_us_per_step": busy,
            "_wall_s_per_step": wall,
        }

    def test_delta_computed_when_a_and_d2_both_valid(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        lora = legs[0]["chains"][attribute.CHAIN_LORA]
        self.assertEqual(lora["status"], "measured_via_d2_delta")
        self.assertAlmostEqual(lora["gpu_busy_us"], 10000.0, places=6)
        self.assertAlmostEqual(lora["share_gpu_busy"], 10000.0 / 60000.0, places=9)
        # D2 itself never gets a C-LORA delta of its own.
        self.assertNotIn(attribute.CHAIN_LORA, legs[1]["chains"])

    def test_delta_absent_without_a_matching_d2_leg(self):
        legs = [self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0)]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_delta_absent_when_d2_leg_is_invalid(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_INVALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_bf16_a_leg_never_gets_a_delta_d2_is_f32_only(self):
        legs = [
            self._leg("clip-text-A2", "clip-text", "A", attribute.VERDICT_VALID, 60000.0, dtype="bf16"),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0, dtype="f32"),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_d1_minus_d2_advisory_attached_when_all_three_present(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D1", "clip-text", "D1", attribute.VERDICT_VALID, 55000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertAlmostEqual(legs[0]["d1_minus_d2_busy_us_advisory"], 5000.0, places=6)


def _write_leg_dir(base: Path, leg_id: str, manifest_extra: dict, census: dict) -> Path:
    leg_dir = base / leg_id
    leg_dir.mkdir(parents=True)
    manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
    manifest.update(manifest_extra)
    (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (leg_dir / "census.json").write_text(json.dumps(census), encoding="utf-8")
    return leg_dir


class AttributeLegAndReportTests(unittest.TestCase):
    def test_end_to_end_on_the_real_fixture_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
            self.assertEqual(report["summary"]["legs_total"], 1)
            self.assertEqual(report["summary"]["legs_valid"], 1)
            row = report["legs"][0]
            self.assertEqual(row["verdict"], attribute.VERDICT_VALID)
            self.assertNotIn("_role", row)
            self.assertNotIn("_gpu_busy_us_per_step", row)
            for name in (
                attribute.CHAIN_LN,
                attribute.CHAIN_GELU,
                attribute.chain_attn("clip-text"),
                attribute.CHAIN_LORA,
            ):
                self.assertIn(name, row["chains"])
            self.assertEqual(row["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_missing_manifest_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "legs" / "clip-text-A1"
            leg_dir.mkdir(parents=True)
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)
            self.assertTrue(row["reasons"])

    def test_missing_census_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "legs" / "clip-text-A1"
            leg_dir.mkdir(parents=True)
            manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
            (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)

    def test_excluded_from_chain_attribution_is_invalid_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            census = load_fixture_census()
            census["excluded_from_chain_attribution"] = True
            leg_dir = _write_leg_dir(Path(tmp) / "legs", "clip-text-A1", {}, census)
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)
            self.assertTrue(any("excluded_from_chain_attribution" in r for r in row["reasons"]))

    def test_census_not_ok_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = _write_leg_dir(
                Path(tmp) / "legs",
                "clip-text-A1",
                {"census_ok": False, "census_exit": 1},
                load_fixture_census(),
            )
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)

    def test_two_legs_a_and_d2_compose_a_full_report_with_lora_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            d2_census = load_fixture_census()
            # A cheaper synthetic D2 twin: same shapes, less total busy (the
            # `lora_linear_fused` kernel forced eager and removed) — kept
            # structurally real (same 15 rows) so `attribute_census` runs
            # the identical mapping; only the aggregate is perturbed to
            # exercise the delta arithmetic without inventing a new export.
            d2_census["gpu_kernel_us_per_step"] = load_fixture_census()["gpu_kernel_us_per_step"] - 5000.0
            _write_leg_dir(
                legs_dir,
                "clip-text-D2",
                {"kernels_disabled": ["lora_linear_fused"]},
                d2_census,
            )
            report = attribute.build_report(legs_dir)
            self.assertEqual(report["summary"]["legs_valid"], 2)
            by_id = {row["leg_id"]: row for row in report["legs"]}
            lora = by_id["clip-text-A1"]["chains"][attribute.CHAIN_LORA]
            self.assertEqual(lora["status"], "measured_via_d2_delta")
            self.assertAlmostEqual(lora["gpu_busy_us"], 5000.0, places=6)
            self.assertNotIn(attribute.CHAIN_LORA, by_id["clip-text-D2"]["chains"])


class TableFormattingTests(unittest.TestCase):
    def test_format_table_lists_every_chain_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
            table = attribute.format_table(report)
            self.assertIn("clip-text-A1", table)
            self.assertIn(attribute.CHAIN_LN, table)
            self.assertIn(attribute.chain_attn("clip-text"), table)


class HtsatStageSignatureSmokeTests(unittest.TestCase):
    """Forward-compatibility smoke test only — no real HTSAT export exists
    to verify these declared constants against yet (module doc)."""

    def test_four_stages_with_positive_window_and_heads(self):
        stages = attribute.htsat_stage_signatures()
        self.assertEqual(len(stages), 4)
        for stage in stages:
            self.assertGreater(stage["depth"], 0)
            self.assertGreater(stage["heads"], 0)
            self.assertGreater(stage["window_size"], 0)


class CliEndToEndTests(unittest.TestCase):
    def test_main_writes_a_report_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            out_path = Path(tmp) / "out.json"
            result = subprocess.run(
                [sys.executable, str(ATTRIBUTE), "--legs-dir", str(legs_dir), "--out", str(out_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["legs_valid"], 1)
            self.assertIn("clip-text-A1", result.stderr)

    def test_main_refuses_a_missing_legs_dir(self):
        result = subprocess.run(
            [sys.executable, str(ATTRIBUTE), "--legs-dir", "/nonexistent/path/xyz"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
