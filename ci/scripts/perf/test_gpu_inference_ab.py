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
        "row_count": 256,
        "warmup": 2,
        "corpus_sha256": "0" * 64,
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
    def test_aa_shaped_fixture_ratio_is_near_1_and_advisory_within_band(self):
        """An A/A-shaped fixture (every leg measures the SAME embed p50 —
        the shape `--aa-null` produces when nothing regressed) must merge
        GREEN with `combined_embed_p50_ratio` ~= 1.0 and
        `advisory=within_placeholder_band`.
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
        self.assertEqual(merged["advisory"]["classification"], "within_placeholder_band")
        self.assertTrue(merged["advisory"]["band_not_pre_registered"])

    def test_degraded_fixture_is_advisory_outside_band_but_still_exits_0(self):
        """A leg set where the `b`-role legs are genuinely slower (a real
        regression shape) classifies `advisory=outside_placeholder_band` in
        the printed row — but v1 is recording-only, so the exit code stays
        0 (GREEN): a premise-clean run with an unfavorable ratio is never
        itself a refusal. `classify_advisory` deliberately never returns
        "pass"/"fail" (round-1 adversarial audit advisory: those words read
        as a gate verdict, which this v1 instrument is not).
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
        self.assertEqual(merged["advisory"]["classification"], "outside_placeholder_band")

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

    def test_a_parent_leg_missing_identity_fields_entirely_is_neutral_not_invalid(self):
        """round-1 adversarial audit B3: a PARENT leg built before issue
        #335's own identity contract landed simply cannot EMIT a field this
        older binary never knew to record — its report JSON is missing the
        key entirely, never present-but-different. This is NOT the same
        claim as "the two legs proved they ran a different premise" (that
        earns INVALID/1 above); it earns the neutral INCOMPLETE_IDENTITY/75
        instead.
        """
        a1 = gpu_inference_tier(embed_p50_ms=8.0)
        del a1["row_count"]
        del a1["corpus_sha256"]
        a2 = gpu_inference_tier(embed_p50_ms=8.0)
        del a2["row_count"]
        del a2["corpus_sha256"]

        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", a1)
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", a2)

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 75)
        self.assertEqual(merged["status"], "INCOMPLETE_IDENTITY")
        self.assertTrue(merged["leg_premise_violations"])
        self.assertTrue(all("missing from" in v for v in merged["leg_premise_violations"]))
        self.assertIn("incomplete_identity_reason", merged)

    def test_a_missing_field_never_masks_a_genuine_value_divergence(self):
        """A single "differs:"-shaped violation among an otherwise
        ALL-missing-field violation set must still promote the WHOLE
        refusal to INVALID/1 — a real divergence is never masked by an
        also-missing field elsewhere landing the merge in the softer
        neutral bucket instead.
        """
        a1 = gpu_inference_tier(embed_p50_ms=8.0)
        del a1["row_count"]
        b1 = gpu_inference_tier(embed_p50_ms=8.0, compute_precision="bf16")  # a genuine VALUE divergence

        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", a1)
            write_leg(raw_dir, "b1", b1)
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 1)
        self.assertEqual(merged["status"], "INVALID")
        self.assertTrue(any("differs:" in v for v in merged["leg_premise_violations"]))


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


class InvalidMeasurementTests(unittest.TestCase):
    """round-1 adversarial audit advisory: an identity-CLEAN leg set can
    still carry a malformed MEASUREMENT (a `null` `p50_ms.value`, a zero
    baseline) — `_measurement_value`/`adjacent_pair_ratio` deliberately
    raise on this rather than silently substituting a placeholder;
    `build_report` must catch that and write a typed refusal (never an
    uncaught traceback with no report at all).
    """

    def test_a_null_embed_p50_ms_value_is_a_typed_refusal_not_a_crash(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=None))
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            # Must not raise -- the whole point of this test.
            merged, exit_code = gpu_inference_ab.build_report(raw_dir)

        self.assertEqual(exit_code, 1)
        self.assertEqual(merged["status"], "INVALID_MEASUREMENT")
        self.assertIn("invalid_measurement_reason", merged)
        self.assertIn("ZeroDivisionError", merged["invalid_measurement_reason"])

    def test_main_writes_a_report_even_on_a_measurement_defect(self):
        """The report-writing contract holds even on this refusal path —
        `main()` (the real CLI entry point) must still write a JSON report
        and a table file, never crash with nothing on disk.
        """
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            write_leg(raw_dir, "a1", gpu_inference_tier(embed_p50_ms=None))
            write_leg(raw_dir, "b1", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "b2", gpu_inference_tier(embed_p50_ms=8.0))
            write_leg(raw_dir, "a2", gpu_inference_tier(embed_p50_ms=8.0))

            rc = gpu_inference_ab.main([raw_dir, out_dir])

            self.assertEqual(rc, 1)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "gpu_inference_ab_report.json")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "gpu_inference_ab_table.txt")))


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


def multiplicative_drift(true_value, k, t):
    """`true_value * (1 + k*t)` -- the synthetic MULTIPLICATIVE linear drift
    model [`DriftCancellationTests`] injects: the physically relevant
    clock/thermal model (a GPU that throttles increasingly over a run
    scales EVERY measurement's wall-time by a growing FACTOR, not by a
    fixed absolute offset)."""
    return true_value * (1 + k * t)


class DriftCancellationTests(unittest.TestCase):
    """Round-1 adversarial audit B4: an earlier version of
    `gpu_inference_ab.py`'s own module doc claimed adjacent-pair averaging
    ([`gpu_inference_ab.combined_embed_p50_ratio`]) was a SUPERIOR estimator
    to a naive mean-of-all-A-vs-mean-of-all-B one -- that claim was FALSE.
    Under a MULTIPLICATIVE linear drift model, the A,B,B,A leg ORDER is
    what cancels the first-order drift term (placing the two B-role legs
    symmetrically BETWEEN the two A-role legs equalizes each role's own
    MEAN measurement time), and BOTH combining conventions are unbiased to
    first order under that order -- adjacent-pairing is a REPORTING
    convention (it additionally surfaces two per-pair ratios for
    diagnostic visibility), not a smaller-bias estimator. This class proves
    the corrected claim mechanically: the real production estimator
    recovers the true ratio to first order under the A,B,B,A order it
    actually runs, and the SAME drift would NOT have cancelled under an
    A,A,B,B order (an order this producer never actually uses -- see
    `gpu_inference_ab.sh`'s own header).
    """

    def test_multiplicative_drift_cancels_to_first_order_under_the_real_a_b_b_a_order(self):
        a_true, b_true = 10.0, 12.0
        true_ratio = b_true / a_true
        k = 0.01  # 1%-per-unit-time drift -- small enough that O(k) vs O(k**2) is a meaningful, discriminating comparison
        t1, t2, t3, t4 = 1.0, 2.0, 3.0, 4.0  # a1, b1, b2, a2's own measurement times, in RUN order

        legs = {
            "a1": gpu_inference_tier(embed_p50_ms=multiplicative_drift(a_true, k, t1)),
            "b1": gpu_inference_tier(embed_p50_ms=multiplicative_drift(b_true, k, t2)),
            "b2": gpu_inference_tier(embed_p50_ms=multiplicative_drift(b_true, k, t3)),
            "a2": gpu_inference_tier(embed_p50_ms=multiplicative_drift(a_true, k, t4)),
        }
        # Drives the REAL production estimator -- never a re-implementation
        # of its own math.
        ratio, pair_ratios = gpu_inference_ab.combined_embed_p50_ratio(legs)

        # First-order cancellation: the residual error must be O(k**2) --
        # MUCH smaller than the O(k) error either single UNPAIRED
        # measurement's own drift would show on its own (roughly
        # `true_ratio * k * (t2 - t1)` = 0.01*12 = 0.12 here, an order of
        # magnitude above this envelope).
        self.assertLess(
            abs(ratio - true_ratio),
            (k**2) * 100,
            f"combined_embed_p50_ratio must recover the true ratio to first order under a multiplicative "
            f"drift model and the REAL A,B,B,A order: got {ratio}, true {true_ratio}, k={k}, "
            f"pair_ratios={pair_ratios}",
        )
        # Neither individual pair ratio need be unbiased on its own -- only
        # their MEAN is (the whole point of averaging two OPPOSITELY-biased
        # pairs) -- so this test does not assert anything about
        # `pair_ratios` individually, only about the combined estimator.

    def test_an_a_a_b_b_order_would_not_have_cancelled_the_same_drift(self):
        """Order sanity (round-1 adversarial audit B4): the SAME
        multiplicative drift, injected as if the legs had run in A, A, B, B
        order instead -- a shape `gpu_inference_ab.sh` never actually
        produces (it always runs A, B, B, A, see that script's own header)
        -- shows a FIRST-ORDER bias the real A,B,B,A order avoids. Proves
        the cancellation this design relies on is bought by the ORDER
        itself, not merely by averaging two numbers together regardless of
        when they were measured.
        """
        a_true, b_true = 10.0, 12.0
        true_ratio = b_true / a_true
        k = 0.01
        t1, t2, t3, t4 = 1.0, 2.0, 3.0, 4.0  # physical run order: A, A, B, B

        a1_val = multiplicative_drift(a_true, k, t1)
        a2_val = multiplicative_drift(a_true, k, t2)
        b1_val = multiplicative_drift(b_true, k, t3)
        b2_val = multiplicative_drift(b_true, k, t4)
        # An A,A,B,B run has no leg pair that straddles an A and a B at
        # all, so there is no "adjacent-pair" analog to compute here in the
        # first place -- the natural comparison this run order WOULD
        # produce is mean(B)/mean(A), computed directly (never through
        # `combined_embed_p50_ratio`, which assumes the real A,B,B,A
        # leg-role shape this order does not have).
        naive_ratio = ((b1_val + b2_val) / 2) / ((a1_val + a2_val) / 2)

        self.assertGreater(
            abs(naive_ratio - true_ratio),
            k / 4,
            "an A,A,B,B run order was expected to show a first-order (O(k)) bias under this SAME "
            "multiplicative drift -- if this assertion fails, the order-balancing rationale itself is "
            "wrong, not merely mis-stated in the module doc",
        )


if __name__ == "__main__":
    unittest.main()
