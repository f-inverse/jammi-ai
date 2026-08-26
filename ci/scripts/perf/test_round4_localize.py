#!/usr/bin/env python3
"""Tests for `round4_localize.py` -- esc-045 round 4/5's 84-row cos/relerr
comparator (A3, GH#374 phase-4 audit).

Runs the aggregation-math suite twice (numpy-available and forced
pure-Python fallback), mirroring `test_compare_grad_oracle.py`'s own
dependency-injection convention, and separately proves the hand-rolled
safetensors parser (`load_f32_dump`) round-trips a byte-for-byte
hand-constructed safetensors blob correctly -- no `safetensors` package
needed anywhere in this file.

Run directly: `python3 ci/scripts/perf/test_round4_localize.py`
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_grad_oracle as cgo  # noqa: E402
import round4_localize as r4l  # noqa: E402


def write_safetensors(path: str, tensors: dict[str, list[float]]) -> None:
    """Hand-writes a minimal, spec-conformant safetensors file (F32 only)
    -- the producer this test uses to prove `load_f32_dump` parses the
    REAL format, not a private test-only shape."""
    header: dict = {}
    blobs: list[bytes] = []
    offset = 0
    for name, values in tensors.items():
        raw = struct.pack(f"<{len(values)}f", *values)
        header[name] = {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for b in blobs:
            fh.write(b)


class SafetensorsRoundTripTests(unittest.TestCase):
    def test_round_trips_multiple_tensors(self):
        tensors = {
            "boundary.1": [1.0, -2.5, 3.25],
            "qkv.0": [0.0, 0.125],
            "mlp_input.0": [7.5],
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dump.safetensors")
            write_safetensors(path, tensors)
            loaded = r4l.load_f32_dump(path)
        self.assertEqual(set(loaded.keys()), set(tensors.keys()))
        for name, values in tensors.items():
            for a, b in zip(loaded[name], values):
                self.assertAlmostEqual(a, b, places=5)

    def test_empty_tensor_round_trips_to_empty_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dump.safetensors")
            write_safetensors(path, {"boundary.1": []})
            loaded = r4l.load_f32_dump(path)
        self.assertEqual(loaded["boundary.1"], [])

    def test_non_f32_dtype_is_a_typed_refusal(self):
        header = {
            "boundary.1": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dump.safetensors")
            with open(path, "wb") as fh:
                fh.write(struct.pack("<Q", len(header_bytes)))
                fh.write(header_bytes)
                fh.write(b"\x00\x00")
            with self.assertRaises(ValueError):
                r4l.load_f32_dump(path)

    def test_truncated_header_is_a_typed_refusal_not_a_crash(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dump.safetensors")
            with open(path, "wb") as fh:
                fh.write(struct.pack("<Q", 1000))  # claims 1000 header bytes
                fh.write(b"{}")  # far fewer actually present
            with self.assertRaises(ValueError):
                r4l.load_f32_dump(path)


class AggregationMathMixin:
    """Shared test bodies, run under BOTH numpy and pure-Python fallback
    (see the two concrete subclasses below) -- family F's "arm parity"
    convention (`compare_grad_oracle.py`'s own
    `ArmParityTests`/`ComparatorMathTests*` pattern), so a bug reachable
    ONLY on the fallback path (the common case in a numpy-less CI image)
    is not silently invisible to a dev machine that happens to have numpy.
    """

    def test_matched_keys_is_the_intersection_not_a_union(self):
        arms = {
            "a": {"x": [1.0], "y": [2.0]},
            "b": {"x": [1.0], "z": [3.0]},
        }
        truth = {"x": [1.0], "y": [2.0], "z": [3.0]}
        self.assertEqual(r4l.matched_keys(arms, truth), ["x"])

    def test_cosine_one_for_identical_vectors_via_shared_helper(self):
        arms = {"a": {"boundary.1": [1.0, 2.0, 3.0]}}
        truth = {"boundary.1": [1.0, 2.0, 3.0]}
        rows = r4l.per_tensor_cosines(arms, truth)
        self.assertAlmostEqual(rows["boundary.1"]["a"], 1.0, places=6)

    def test_cosine_minus_one_for_opposite_vectors(self):
        arms = {"a": {"boundary.1": [-1.0, -2.0, -3.0]}}
        truth = {"boundary.1": [1.0, 2.0, 3.0]}
        rows = r4l.per_tensor_cosines(arms, truth)
        self.assertAlmostEqual(rows["boundary.1"]["a"], -1.0, places=6)

    def test_cosine_zero_for_orthogonal_vectors(self):
        arms = {"a": {"boundary.1": [1.0, 0.0]}}
        truth = {"boundary.1": [0.0, 1.0]}
        rows = r4l.per_tensor_cosines(arms, truth)
        self.assertAlmostEqual(rows["boundary.1"]["a"], 0.0, places=6)

    def test_relerr_zero_when_arm_equals_truth(self):
        arms = {"a": {"boundary.1": [1.0, 2.0], "qkv.0": [3.0]}}
        truth = {"boundary.1": [1.0, 2.0], "qkv.0": [3.0]}
        relerr = r4l.concatenated_relerr(arms, truth)
        self.assertAlmostEqual(relerr["a"], 0.0, places=6)

    def test_relerr_matches_hand_computed_value(self):
        # truth = [3, 4] (norm 5); arm = [3, 0] -> diff = [0, -4], norm 4.
        # relerr = 4 / 5 = 0.8, hand-verifiable independent of any library.
        arms = {"a": {"boundary.1": [3.0, 0.0]}}
        truth = {"boundary.1": [3.0, 4.0]}
        relerr = r4l.concatenated_relerr(arms, truth)
        self.assertAlmostEqual(relerr["a"], 0.8, places=6)

    def test_relerr_concatenation_order_is_sorted_key_deterministic(self):
        # Two arms with the SAME per-tensor values but built from dicts
        # constructed in a DIFFERENT insertion order must still agree
        # exactly -- concatenation order is derived from sorted(keys), not
        # dict iteration order (family J: fixed fold order).
        truth = {"qkv.0": [1.0, 2.0], "boundary.1": [3.0, 4.0]}
        arm_a = {"qkv.0": [1.5, 2.5], "boundary.1": [3.5, 4.5]}
        arm_b = {"boundary.1": [3.5, 4.5], "qkv.0": [1.5, 2.5]}
        relerr_a = r4l.concatenated_relerr({"x": arm_a}, truth)["x"]
        relerr_b = r4l.concatenated_relerr({"x": arm_b}, truth)["x"]
        self.assertAlmostEqual(relerr_a, relerr_b, places=10)

    def test_relerr_all_zero_truth_uses_norm_floor_not_a_zero_division(self):
        arms = {"a": {"boundary.1": [1.0, 1.0]}}
        truth = {"boundary.1": [0.0, 0.0]}
        relerr = r4l.concatenated_relerr(arms, truth)
        self.assertTrue(math.isfinite(relerr["a"]))
        self.assertGreater(relerr["a"], 0.0)

    def test_band_means_splits_low_and_high_by_layer_index(self):
        rows = {
            "boundary.1": {"a": 0.2},
            "boundary.5": {"a": 0.3},
            "boundary.18": {"a": 0.9},
            "boundary.27": {"a": 0.95},
        }
        bands = r4l.band_means(rows, ["a"], band_split=18)
        self.assertAlmostEqual(bands["boundary"]["low"]["a"], 0.25, places=6)  # (0.2+0.3)/2
        self.assertAlmostEqual(bands["boundary"]["high"]["a"], 0.925, places=6)  # (0.9+0.95)/2

    def test_band_means_empty_band_is_nan_not_a_crash(self):
        rows = {"boundary.20": {"a": 0.9}}
        bands = r4l.band_means(rows, ["a"], band_split=18)
        self.assertTrue(math.isnan(bands["boundary"]["low"]["a"]))

    def test_build_report_end_to_end_on_a_synthetic_84_row_shaped_fixture(self):
        # A small (3-layer, not 28) stand-in for the real ModernBERT-large
        # shape -- proves the FULL pipeline (matched_keys -> cosines ->
        # relerr -> bands) composes correctly, not merely each piece in
        # isolation.
        num_layers = 3
        truth = {}
        fused = {}
        eager = {}
        for i in range(num_layers):
            truth[f"boundary.{i}"] = [float(i + 1), float(i + 2)]
            # `fused` matches truth exactly (cos=1, relerr=0).
            fused[f"boundary.{i}"] = [float(i + 1), float(i + 2)]
            # `eager` is the negated vector (cos=-1) for every layer.
            eager[f"boundary.{i}"] = [-float(i + 1), -float(i + 2)]
        report = r4l.build_report({"fused": fused, "eager": eager}, truth, band_split=1)
        self.assertEqual(sorted(report["arm_names"]), ["eager", "fused"])
        for key in report["keys"]:
            self.assertAlmostEqual(report["cos_per_tensor"][key]["fused"], 1.0, places=6)
            self.assertAlmostEqual(report["cos_per_tensor"][key]["eager"], -1.0, places=6)
        self.assertAlmostEqual(report["all_matched_relerr"]["fused"], 0.0, places=6)
        self.assertGreater(report["all_matched_relerr"]["eager"], 1.0)  # opposite vectors: large relerr

    def test_build_report_refuses_when_no_keys_are_matched(self):
        with self.assertRaises(ValueError):
            r4l.build_report({"a": {"x": [1.0]}}, {"y": [1.0]}, band_split=18)

    def test_format_table_contains_every_arm_and_every_key(self):
        report = r4l.build_report(
            {"fused": {"boundary.0": [1.0]}}, {"boundary.0": [1.0]}, band_split=18
        )
        text = r4l.format_table(report)
        self.assertIn("fused", text)
        self.assertIn("boundary.0", text)


class AggregationMathWhateverNumpyIsAvailable(AggregationMathMixin, unittest.TestCase):
    """Whatever this process's numpy availability is (`r4l.HAVE_NUMPY` at
    import time)."""


class AggregationMathForcedPureFallback(AggregationMathMixin, unittest.TestCase):
    """Forces BOTH `round4_localize.HAVE_NUMPY` (this file's own parsing/
    relerr code) AND `compare_grad_oracle.HAVE_NUMPY` (the imported
    `cosine_similarity`/`_dot`/`_norm` this file calls, which look up
    THEIR OWN module's `HAVE_NUMPY` global regardless of which module
    calls them) to `False` for the duration of each test -- forcing the
    pure-Python fallback path on BOTH modules at once, the only way to
    genuinely exercise `round4_localize`'s fallback end to end."""

    def setUp(self):
        self._orig_r4l = r4l.HAVE_NUMPY
        self._orig_cgo = cgo.HAVE_NUMPY
        r4l.HAVE_NUMPY = False
        cgo.HAVE_NUMPY = False

    def tearDown(self):
        r4l.HAVE_NUMPY = self._orig_r4l
        cgo.HAVE_NUMPY = self._orig_cgo


class MainEntryPointTests(unittest.TestCase):
    def test_main_end_to_end_with_real_files(self):
        with tempfile.TemporaryDirectory() as d:
            truth_path = os.path.join(d, "truth.safetensors")
            fused_path = os.path.join(d, "fused.safetensors")
            out_path = os.path.join(d, "report.json")
            write_safetensors(truth_path, {"boundary.0": [1.0, 2.0]})
            write_safetensors(fused_path, {"boundary.0": [1.0, 2.0]})
            rc = r4l.main(
                [
                    "--truth",
                    truth_path,
                    "--arm",
                    f"fused={fused_path}",
                    "--out",
                    out_path,
                ]
            )
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                report = json.load(fh)
        self.assertAlmostEqual(report["cos_per_tensor"]["boundary.0"]["fused"], 1.0, places=6)

    def test_main_refuses_a_malformed_arm_spec(self):
        with tempfile.TemporaryDirectory() as d:
            truth_path = os.path.join(d, "truth.safetensors")
            write_safetensors(truth_path, {"boundary.0": [1.0]})
            rc = r4l.main(["--truth", truth_path, "--arm", "not-a-key-value-pair"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
