#!/usr/bin/env python3
"""Self-test for `gpu_inference_ab.py` (issue #335) — the `test_ab_merge.py`
style: drives the REAL `build_report`/`main` entry points against fixture
leg directories shaped exactly like `gpu_inference_ab.sh`'s own
`.exit`/`.json` output, never a hand-rolled call into an inner helper with
literal tuples standing in for a report.

Stdlib-only (`unittest`), same footing every other `ci/scripts/perf/test_*.py`
in this directory takes.

Run: `python3 ci/scripts/perf/test_gpu_inference_ab.py`
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_inference_ab  # noqa: E402
from identity_fields import GPU_INFERENCE_IDENTITY_FIELDS  # noqa: E402


def measurement(value, unit):
    return {"value": value, "unit": unit}


def gpu_lane(rows=8, rows_per_s=1000.0, p50_ms=8.0, p99_ms=9.0, deterministic=True):
    return {
        "rows": rows,
        "rows_per_s": measurement(rows_per_s, "rows_per_s"),
        "p50_ms": measurement(p50_ms, "ms"),
        "p99_ms": measurement(p99_ms, "ms"),
        "deterministic": deterministic,
    }


def gpu_inference_tier(embed_p50_ms=8.0, infer_p50_ms=6.0, **identity_overrides):
    """A fixture `GpuInferenceTier`-shaped dict carrying every declared
    identity + provenance field, correctly populated by default —
    `**identity_overrides` perturbs any named field (identity mismatch /
    degraded-leg fixtures below).
    """
    tier = {
        "device": "cuda:0",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "iters": 20,
        "corpus_seed": 0,
        "warmup": 2,
        "compute_precision": "f32",
        "embed_checkpoint_config_sha256": "a" * 64,
        "embed_checkpoint_weights_sha256": "b" * 64,
        "embed_checkpoint_tokenizer_sha256": "c" * 64,
        "infer_checkpoint_config_sha256": "d" * 64,
        "infer_checkpoint_weights_sha256": "e" * 64,
        "infer_checkpoint_tokenizer_sha256": "f" * 64,
        "kernels_disabled_requested": [],
        "flash_compiled": True,
        "build_features": ["cuda"],
        "embed": gpu_lane(p50_ms=embed_p50_ms),
        "infer": gpu_lane(p50_ms=infer_p50_ms),
    }
    tier.update(identity_overrides)
    return tier


def write_leg(raw_dir, name, tier=None, exit_code="0", build_sha=None):
    if tier is None:
        # A FAIL leg: exit file present and nonzero, no valid report.
        with open(os.path.join(raw_dir, f"{name}.exit"), "w", encoding="utf-8") as fh:
            fh.write(exit_code)
        with open(os.path.join(raw_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            fh.write("")
        return
    report = {
        "tiers": {"gpu_inference": tier},
        "provenance": {"build_sha": build_sha or ("0" * 40)},
    }
    with open(os.path.join(raw_dir, f"{name}.exit"), "w", encoding="utf-8") as fh:
        fh.write(exit_code)
    import json

    with open(os.path.join(raw_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh)


def write_all_ok_legs(raw_dir, embed_p50_by_leg, **shared_overrides):
    """Write all four [`gpu_inference_ab.LEG_ORDER`] legs OK, each leg's
    `embed.p50_ms` taken from `embed_p50_by_leg` (a `{leg_name: value}`
    dict), every other field shared/identical (a clean, identity-agreeing
    A/B/A/B set) unless `shared_overrides` perturbs one.
    """
    for name in gpu_inference_ab.LEG_ORDER:
        tier = gpu_inference_tier(embed_p50_ms=embed_p50_by_leg[name], **shared_overrides)
        write_leg(raw_dir, name, tier)


class BuildReportGreenPathTests(unittest.TestCase):
    def test_aa_shaped_fixture_ratio_is_near_1_and_advisory_pass(self):
        """An A/A-shaped fixture (every leg measures the SAME embed p50 —
        the shape `--aa-null` produces when nothing regressed) must merge
        GREEN with `combined_embed_p50_ratio` ~= 1.0 and `advisory=pass`.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_all_ok_legs(
                raw_dir,
                {"a1": 8.0, "b1": 8.0, "b2": 8.0, "a2": 8.0},
            )
            merged, exit_code = gpu_inference_ab.build_report(raw_dir, a_sha="a" * 40, b_sha="b" * 40)

        self.assertEqual(exit_code, 0)
        self.assertEqual(merged["status"], "GREEN")
        self.assertAlmostEqual(merged["combined_embed_p50_ratio"], 1.0, places=9)
        self.assertEqual(merged["advisory"]["classification"], "pass")
        self.assertTrue(merged["advisory"]["band_not_pre_registered"])

    def test_degraded_fixture_is_advisory_fail_but_still_exits_0(self):
        """A leg set where the `b`-role legs are genuinely slower (a real
        regression shape) classifies `advisory=fail` in the printed row —
        but v1 is recording-only, so the exit code stays 0 (GREEN): a
        premise-clean run with an unfavorable ratio is never itself a
        refusal.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            # b/a ratio = 16/8 = 2.0 for both pairs -- well outside the
            # placeholder [0.90, 1.10] band.
            write_all_ok_legs(
                raw_dir,
                {"a1": 8.0, "b1": 16.0, "b2": 16.0, "a2": 8.0},
            )
            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 0, "a degraded ratio must never itself fail the merge (v1 recording-only)")
        self.assertEqual(merged["status"], "GREEN")
        self.assertAlmostEqual(merged["combined_embed_p50_ratio"], 2.0, places=9)
        self.assertEqual(merged["advisory"]["classification"], "fail")

    def test_adjacent_pair_ratios_are_computed_in_b_over_a_orientation_regardless_of_leg_name(self):
        """`adjacent_pair_ratio` must read `b/a` for BOTH pairs even though
        the physical run order differs: pair one is (a1, b1) — a physically
        ran first; pair two is (b2, a2) — b physically ran first. Both
        pairs' printed ratios must still be `b-role / a-role`, never
        "whichever leg came first in the pair".
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_all_ok_legs(
                raw_dir,
                {"a1": 10.0, "b1": 12.0, "b2": 9.0, "a2": 10.0},
            )
            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 0)
        self.assertAlmostEqual(merged["adjacent_pair_ratios"]["a1/b1"], 12.0 / 10.0, places=9)
        self.assertAlmostEqual(merged["adjacent_pair_ratios"]["b2/a2"], 9.0 / 10.0, places=9)
        self.assertAlmostEqual(
            merged["combined_embed_p50_ratio"],
            ((12.0 / 10.0) + (9.0 / 10.0)) / 2.0,
            places=9,
        )


class IdentityMismatchTests(unittest.TestCase):
    def test_identity_mismatch_between_legs_is_invalid_and_nonzero_exit(self):
        """A `b`-role leg whose `compute_precision` differs from the `a`-role
        legs' must refuse (INVALID, nonzero exit) — the two legs did not
        measure the same premise, so their ratio is meaningless.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0, compute_precision="bf16"))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 1)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(merged["leg_premise_violations"])
        self.assertTrue(any("compute_precision" in v for v in merged["leg_premise_violations"]))

    def test_checkpoint_sha_mismatch_is_invalid(self):
        """A PR that touches the committed embed fixture bytes moves
        `embed_checkpoint_weights_sha256` — this MUST be caught as a premise
        mismatch, not silently averaged into the ratio as if the two legs
        served the same checkpoint.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(
                raw_dir,
                "b1",
                gpu_inference_tier(embed_p50_ms=8.0, embed_checkpoint_weights_sha256="9" * 64),
            )
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 1)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(any("embed_checkpoint_weights_sha256" in v for v in merged["leg_premise_violations"]))


class MissingLegTests(unittest.TestCase):
    def test_one_missing_leg_is_neutral_exit_75_never_fail(self):
        """A missing leg (no `.exit` file at all — e.g. the pod died before
        that leg ran) must be a NEUTRAL "nothing to compare" (exit 75), never
        treated as a hard FAIL — mirrors `runpod_gpu_prove.sh`'s own
        no-capacity 75 convention.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            # a2 never written at all -- MISSING.

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 75)
        self.assertEqual(merged["status"], "INCOMPLETE")
        self.assertEqual(merged["missing_legs"], ["a2"])

    def test_one_failed_leg_is_also_neutral_exit_75(self):
        """A leg that ran but exited nonzero (a real build/serve failure,
        e.g. no CUDA device on this pod) is likewise INCOMPLETE/75, never a
        distinct hard-FAIL exit code — a leg failure and a leg's total
        absence are the same "could not compare" state from this merger's
        own point of view.
        """
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b1", tier=None, exit_code="1")  # FAIL leg
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 75)
        self.assertEqual(merged["status"], "INCOMPLETE")
        self.assertEqual(merged["missing_legs"], ["b1"])

    def test_empty_raw_dir_is_neutral_75_all_four_legs_missing(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 75)
        self.assertEqual(merged["status"], "INCOMPLETE")
        self.assertEqual(set(merged["missing_legs"]), set(gpu_inference_ab.LEG_ORDER))


class MainEntryPointTests(unittest.TestCase):
    """Drives `main()` itself (the real CLI entry point `gpu_inference_ab.sh`
    invokes), not just `build_report` — proves the JSON + table are actually
    written to `OUT_DIR` and the process-level exit code matches
    `build_report`'s own.
    """

    def test_main_writes_report_and_table_and_returns_build_report_exit_code(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            write_all_ok_legs(raw_dir, {"a1": 8.0, "b1": 8.0, "b2": 8.0, "a2": 8.0})
            rc = gpu_inference_ab.main([raw_dir, out_dir, "a" * 40, "b" * 40])

            self.assertEqual(rc, 0)
            report_path = os.path.join(out_dir, "gpu_inference_ab_report.json")
            table_path = os.path.join(out_dir, "gpu_inference_ab_table.txt")
            self.assertTrue(os.path.isfile(report_path))
            self.assertTrue(os.path.isfile(table_path))
            with open(table_path, encoding="utf-8") as fh:
                table = fh.read()
            self.assertIn("status=GREEN", table)
            self.assertIn("NOT PRE-REGISTERED", table)

    def test_main_usage_error_on_too_few_args(self):
        rc = gpu_inference_ab.main(["only-one-arg"])
        self.assertEqual(rc, 2)


class IdentityFieldsSharedCoreTests(unittest.TestCase):
    """Proves this module actually calls `ab_merge`'s shared refusal core
    (never a hand-rolled second comparator) -- a field NOT in
    `GPU_INFERENCE_IDENTITY_FIELDS` is never checked, even if it differs.
    """

    def test_a_non_identity_field_difference_is_not_a_violation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=8.0, device_name="box-a"))
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0, device_name="box-b"))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0, device_name="box-b"))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0, device_name="box-a"))

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        # device_name is PROVENANCE, not identity -- differing across legs
        # (two different physical GPUs on the SAME rented pod's own
        # ordinal, or a re-run on a different pod) must not itself
        # invalidate the merge.
        self.assertEqual(exit_code, 0)
        self.assertEqual(merged["status"], "GREEN")

    def test_gpu_inference_identity_fields_is_what_gets_checked(self):
        self.assertEqual(
            set(gpu_inference_ab.leg_identity(gpu_inference_tier()).keys()),
            set(GPU_INFERENCE_IDENTITY_FIELDS),
        )


if __name__ == "__main__":
    unittest.main()
