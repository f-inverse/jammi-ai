#!/usr/bin/env python3
"""`encode_ab.py`, driven through its REAL entry point (`main`) against leg
directories shaped like `encode_ab.sh`'s own `.exit`/`.json` output, and the
pure (torch-free) halves of `torch_encode.py` the two producers must agree on.

Stdlib only: it runs in the CI image's lane.

Run: `python3 ci/scripts/perf/test_encode_ab.py`
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PERF_DIR)
import encode_ab  # noqa: E402
import identity_fields  # noqa: E402

sys.path.insert(0, os.path.join(PERF_DIR, "..", "..", "..", "crates", "jammi-bench", "reference"))
import torch_encode  # noqa: E402

ROWS = (16, 256)
PARTITIONS_N = 4


def identity():
    return {
        "seed": 0,
        "rows": list(ROWS),
        "batch_size": 32,
        "corpus": [
            {"rows": r, "corpus_sha256": f"c{r}", "token_lengths_sha256": f"t{r}", "tokens": 30 * r} for r in ROWS
        ],
        "max_sequence_length": 128,
        "compute_precision": "f32",
        "checkpoint_config_sha256": "cfg",
        "checkpoint_weights_sha256": "w",
        "checkpoint_weights_size_bytes": 102608,
        "checkpoint_tokenizer_sha256": "tok",
        "pooling": "mean",
        "normalize": True,
        "warmup": 2,
        "iters_measured": 10,
        "checkpoint_pooling_sha256": "pool",
        "device_requested": "cpu",
    }


def points(rows_per_s, torch):
    out = []
    for r in ROWS:
        point = {
            "rows": r,
            "rows_per_s": rows_per_s,
            "tokens_per_s": 30.0 * rows_per_s,
            "serve_ms_p50": 1e3 * r / rows_per_s,
            "serve_ms_min": 1e3 * r / rows_per_s,
            "peak_rss_bytes": 2.0e8,
            "peak_vram_delta_bytes": None,
        }
        if torch:
            point["agreement"] = {"rows": r, "cosine_min": 0.999999, "cosine_mean": 0.9999999}
        else:
            point["vectors_digest"] = f"digest{r}"
        out.append(point)
    return out


def leg_report(leg, rows_per_s):
    arm = leg.removesuffix(encode_ab.SECOND_RUN_SUFFIX)
    fit = {"fixed_ms": 3.0, "per_row_ms": 1e3 / rows_per_s, "relative_residual_rms": 0.0}
    if arm in encode_ab.JAMMI_ARMS:
        tier = dict(identity(), partitions=1 if arm == "jammi-p1" else PARTITIONS_N)
        tier.update(points=points(rows_per_s, torch=False), fit_p50=fit, fit_min=fit)
        return {"tool": "jammi-bench", "provenance": {"build_sha": "0" * 40}, "tiers": {"encode_step": tier}}
    tier = {k: v for k, v in identity().items() if k != "seed"}
    tier.update(order=encode_ab.TORCH_ARM_ORDER[arm], ann_index=True)
    tier.update(points=points(rows_per_s, torch=True), fit_p50=fit, fit_min=fit)
    return {"tool": "torch-encode", "provenance": {"torch_version": "2"}, "encode_step": tier}


# rows/s per leg: both jammi runs clear 0.9 x both torch-corpus runs; against
# torch-sorted one pairing clears it and one does not.
RATES = {
    "jammi-p1": 1000.0,
    "jammi-p1-2": 1100.0,
    "jammi-pN": 2000.0,
    "jammi-pN-2": 2100.0,
    "torch-corpus": 1000.0,
    "torch-corpus-2": 1050.0,
    "torch-sorted": 1150.0,
    "torch-sorted-2": 1300.0,
}


def healthy_run():
    return {leg: leg_report(leg, RATES[leg]) for leg in encode_ab.LEG_ORDER}


def merge(reports, markers=None, exits=None):
    """Write `reports` as a raw dir and run the real `main`; returns
    `(merged report, exit code)`."""
    markers = {"partitions": str(PARTITIONS_N), "torch_ann_index": "1", **(markers or {})}
    with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
        for leg, report in reports.items():
            with open(os.path.join(raw_dir, f"{leg}.json"), "w") as fh:
                json.dump(report, fh)
            with open(os.path.join(raw_dir, f"{leg}.exit"), "w") as fh:
                fh.write((exits or {}).get(leg, "0"))
        for name, value in markers.items():
            with open(os.path.join(raw_dir, name), "w") as fh:
                fh.write(value + "\n")
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = encode_ab.main([raw_dir, out_dir, "f" * 40])
        with open(os.path.join(out_dir, "encode_ab_report.json")) as fh:
            return json.load(fh), exit_code


def tier_in(reports, leg):
    report = reports[leg]
    return report["tiers"]["encode_step"] if leg.startswith("jammi") else report["encode_step"]


class LegOrderTests(unittest.TestCase):
    def test_every_arm_runs_twice_centred_on_the_same_instant(self):
        """The palindrome property the drift cancellation rests on: each
        arm's two positions sum to the same number."""
        order = encode_ab.LEG_ORDER
        self.assertEqual(len(order), 2 * len(encode_ab.ARMS))
        centres = {
            arm: order.index(arm) + order.index(arm + encode_ab.SECOND_RUN_SUFFIX) for arm in encode_ab.ARMS
        }
        self.assertEqual(set(centres.values()), {len(order) - 1}, centres)

    def test_a_multiplicative_drift_cancels_out_of_every_pair_of_arms(self):
        """A box whose wall times grow linearly over the run scales every
        arm's mean time alike under this order — and would not under
        A, A, B, B."""
        drift = lambda position: 1.0 + 0.02 * position  # noqa: E731
        true_ms = {"jammi-p1": 10.0, "jammi-pN": 6.0, "torch-corpus": 9.0, "torch-sorted": 5.0}

        def mean_ms(order):
            seen = {}
            for position, leg in enumerate(order):
                arm = leg.removesuffix(encode_ab.SECOND_RUN_SUFFIX)
                seen.setdefault(arm, []).append(true_ms[arm] * drift(position))
            return {arm: sum(v) / len(v) for arm, v in seen.items()}

        balanced = mean_ms(encode_ab.LEG_ORDER)
        blocked = mean_ms([leg for arm in encode_ab.ARMS for leg in encode_ab.runs_of(arm)])
        for a in encode_ab.ARMS:
            for b in encode_ab.ARMS:
                self.assertAlmostEqual(balanced[a] / balanced[b], true_ms[a] / true_ms[b], places=12)
        self.assertNotAlmostEqual(
            blocked["jammi-p1"] / blocked["torch-sorted"], true_ms["jammi-p1"] / true_ms["torch-sorted"], places=2
        )


class GreenRunTests(unittest.TestCase):
    def setUp(self):
        self.merged, self.exit_code = merge(healthy_run())

    def test_a_run_whose_premises_and_outcomes_hold_is_green(self):
        self.assertEqual((self.merged["status"], self.exit_code), ("GREEN", 0), self.merged)
        self.assertEqual(
            sorted(self.merged["comparisons"]),
            sorted(f"{j} vs {t}" for j in encode_ab.JAMMI_ARMS for t in encode_ab.TORCH_ARMS),
        )

    def test_a_rate_ratio_is_the_least_favourable_pairing_of_the_runs(self):
        record = self.merged["comparisons"]["jammi-p1 vs torch-sorted"]["per_rows"]["16"]["rows_per_s"]
        self.assertAlmostEqual(record["conservative"], 1000.0 / 1300.0)
        self.assertAlmostEqual(record["optimistic"], 1100.0 / 1150.0)

    def test_a_cost_ratio_is_least_favourable_the_other_way_up(self):
        record = self.merged["comparisons"]["jammi-p1 vs torch-sorted"]["fit_p50"]["per_row_ms"]
        self.assertAlmostEqual(record["conservative"], (1e3 / 1000.0) / (1e3 / 1300.0))
        self.assertAlmostEqual(record["optimistic"], (1e3 / 1100.0) / (1e3 / 1150.0))

    def test_parity_is_pass_fail_or_indeterminate_by_where_both_estimates_fall(self):
        verdict = lambda name: self.merged["comparisons"][name]["per_rows"]["16"]["parity"]  # noqa: E731
        self.assertEqual(verdict("jammi-p1 vs torch-corpus"), "PASS")  # 1000/1050 >= 0.9
        self.assertEqual(verdict("jammi-p1 vs torch-sorted"), "INDETERMINATE")  # 0.77 .. 0.96
        self.assertEqual(verdict("jammi-pN vs torch-sorted"), "PASS")
        failing, exit_code = merge(healthy_run(), markers={"pass_ratio": "2.5"})
        self.assertEqual(failing["comparisons"]["jammi-pN vs torch-sorted"]["per_rows"]["16"]["parity"], "FAIL")
        self.assertEqual((failing["status"], exit_code), ("GREEN", 0), "a verdict is recorded, never gated")

    def test_a_quantity_a_leg_did_not_measure_has_no_ratio(self):
        self.assertIsNone(
            self.merged["comparisons"]["jammi-p1 vs torch-corpus"]["per_rows"]["16"]["peak_vram_delta_bytes"]
        )


class RefusalTests(unittest.TestCase):
    def assert_refused(self, reports, status, needle, **kwargs):
        merged, exit_code = merge(reports, **kwargs)
        self.assertEqual((merged["status"], exit_code), (status, 1), merged)
        violations = merged["leg_premise_violations"] + merged["outcome_violations"]
        self.assertTrue(any(needle in v for v in violations), violations)
        self.assertEqual(merged["comparisons"], {}, "a refused run records no ratio")

    def test_a_torch_leg_at_another_batch_size_is_invalid(self):
        reports = healthy_run()
        tier_in(reports, "torch-sorted")["batch_size"] = 64
        self.assert_refused(reports, "INVALID", "batch_size")

    def test_a_replicate_under_another_premise_is_invalid(self):
        reports = healthy_run()
        tier_in(reports, "jammi-pN-2")["warmup"] = 0
        self.assert_refused(reports, "INVALID", "warmup")

    def test_a_torch_leg_that_tokenized_differently_is_invalid(self):
        reports = healthy_run()
        for leg in encode_ab.runs_of("torch-corpus"):
            tier_in(reports, leg)["corpus"][0]["token_lengths_sha256"] = "other"
        self.assert_refused(reports, "INVALID", "corpus")

    def test_a_leg_that_contradicts_its_label_is_invalid(self):
        for leg, field, value in (
            ("jammi-pN", "partitions", 1),
            ("torch-sorted-2", "order", "corpus"),
            ("torch-corpus", "ann_index", False),
        ):
            with self.subTest(leg=leg):
                reports = healthy_run()
                tier_in(reports, leg)[field] = value
                self.assert_refused(reports, "INVALID", field)

    def test_a_null_pooling_config_is_a_premise_both_sides_can_share(self):
        reports = healthy_run()
        for leg in reports:
            tier_in(reports, leg)["checkpoint_pooling_sha256"] = None
        self.assertEqual(merge(reports)[0]["status"], "GREEN")
        tier_in(reports, "torch-corpus")["checkpoint_pooling_sha256"] = "pool"
        self.assert_refused(reports, "INVALID", "checkpoint_pooling_sha256")

    def test_partitions_that_changed_the_persisted_vectors_is_an_invalid_measurement(self):
        reports = healthy_run()
        tier_in(reports, "jammi-pN")["points"][1]["vectors_digest"] = "drifted"
        self.assert_refused(reports, "INVALID_MEASUREMENT", "persisted different vectors")

    def test_a_torch_leg_that_embedded_something_else_is_an_invalid_measurement(self):
        reports = healthy_run()
        tier_in(reports, "torch-sorted")["points"][0]["agreement"]["cosine_min"] = 0.42
        self.assert_refused(reports, "INVALID_MEASUREMENT", "under the floor")
        unchecked = healthy_run()
        tier_in(unchecked, "torch-sorted")["points"][0]["agreement"] = None
        self.assert_refused(unchecked, "INVALID_MEASUREMENT", "no jammi vectors")

    def test_a_failed_or_missing_leg_is_incomplete(self):
        merged, exit_code = merge(healthy_run(), exits={"torch-sorted-2": "1"})
        self.assertEqual((merged["status"], exit_code), ("INCOMPLETE", 1))
        self.assertEqual(merged["legs"]["torch-sorted-2"]["outcome"], "FAIL")
        reports = healthy_run()
        del reports["jammi-p1-2"]
        merged, exit_code = merge(reports)
        self.assertEqual((merged["status"], exit_code), ("INCOMPLETE", 1))
        self.assertEqual(merged["legs"]["jammi-p1-2"]["outcome"], "MISSING")

    def test_a_dry_run_is_recorded_as_one(self):
        stubs = {leg: {"tool": "dry-run", "ab_dry_run": True, "leg": leg} for leg in encode_ab.LEG_ORDER}
        merged, exit_code = merge(stubs)
        self.assertEqual((merged["status"], exit_code), ("DRY_RUN", 0))


class SharedDefinitionTests(unittest.TestCase):
    """What the jammi producer and the PyTorch reference must compute alike,
    pinned on the Python side to the same answers `jammi-bench`'s own tests
    pin on the Rust side."""

    def test_the_twin_identity_set_is_the_reference_producers_own(self):
        self.assertEqual(identity_fields.ENCODE_TWIN_IDENTITY_FIELDS, torch_encode.IDENTITY_FIELDS)

    def test_token_lengths_sha256_is_the_pinned_rendering(self):
        # `encode_step.rs::tests::token_lengths_sha256_is_the_pinned_rendering`.
        self.assertEqual(
            torch_encode.token_lengths_sha256([3, 5, 8]),
            "d8c85d93367f9ebb51148c0a399b58ca7386e2de1a4ffcc5e925048f6d4e2250",
        )

    def test_the_cost_fit_recovers_an_exactly_two_term_sweep(self):
        # `timing.rs::tests::an_exactly_two_term_sweep_recovers_both_terms`.
        fit = torch_encode.cost_fit([(r, 7.5 + 0.02 * r) for r in (16, 256, 4096, 16384)])
        self.assertAlmostEqual(fit["fixed_ms"], 7.5, places=9)
        self.assertAlmostEqual(fit["per_row_ms"], 0.02, places=12)
        self.assertLess(fit["relative_residual_rms"], 1e-12)
        self.assertIsNone(torch_encode.cost_fit([(16, 8.0)]))
        self.assertIsNone(torch_encode.cost_fit([(16, 8.0), (16, 8.1)]))

    def test_length_sorted_is_longest_text_first_with_ties_in_input_order(self):
        texts = ["bb", "a", "cccc", "dd"]
        self.assertEqual(torch_encode.forward_order(texts, "corpus"), [0, 1, 2, 3])
        self.assertEqual(torch_encode.forward_order(texts, "length-sorted"), [2, 0, 3, 1])

    def test_a_pooling_declaration_the_engine_refuses_is_refused(self):
        with tempfile.TemporaryDirectory() as model_dir:
            self.assertEqual(torch_encode.resolve_pooling(model_dir), ("mean", None))
            os.makedirs(os.path.join(model_dir, "1_Pooling"))
            path = os.path.join(model_dir, "1_Pooling", "config.json")

            def declare(**flags):
                with open(path, "w") as fh:
                    json.dump(flags, fh)

            declare(pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
            self.assertEqual(torch_encode.resolve_pooling(model_dir)[0], "cls")
            declare(pooling_mode_mean_sqrt_len_tokens=True, pooling_mode_mean_tokens=True)
            self.assertEqual(torch_encode.resolve_pooling(model_dir)[0], "mean")
            for refused in (
                {"pooling_mode_mean_tokens": False},
                {"pooling_mode_lasttoken": True},
                {"pooling_mode_cls_token": True, "pooling_mode_max_tokens": True},
            ):
                declare(**refused)
                with self.assertRaises(ValueError):
                    torch_encode.resolve_pooling(model_dir)

    def test_fold_sweep_refuses_a_point_under_another_premise(self):
        point = lambda rows, batch: {  # noqa: E731
            "rows": [rows],
            "batch_size": batch,
            "corpus": [{"rows": rows}],
            "points": [{"rows": rows, "serve_ms_p50": 1.0 + rows, "serve_ms_min": 1.0 + rows}],
            "fit_p50": None,
            "fit_min": None,
        }
        tier = torch_encode.fold_sweep([point(16, 32), point(256, 32)])
        self.assertEqual(tier["rows"], [16, 256])
        self.assertAlmostEqual(tier["fit_min"]["per_row_ms"], 1.0)
        with self.assertRaises(ValueError):
            torch_encode.fold_sweep([copy.deepcopy(point(16, 32)), point(256, 8)])


if __name__ == "__main__":
    unittest.main()
