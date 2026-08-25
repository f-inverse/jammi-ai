#!/usr/bin/env python3
"""Tests for `compare_grad_oracle.py`.

Runs the FULL suite twice: once against whatever numpy availability this
environment actually has, and once with `ab_merge`-style dependency
injection forcing the pure-Python fallback path (`HAVE_NUMPY = False`) --
this repo's dev/CI environment may or may not have numpy importable
(`torch`'s own transitive dependency, present inside `$TORCH_VENV`, absent
in a bare CI Python), so the fallback path needs its OWN direct coverage,
not just "whatever happened to run today".

Run directly: `python3 ci/scripts/perf/test_compare_grad_oracle.py`
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_grad_oracle as cgo  # noqa: E402


def make_report(loss, gradients):
    """`gradients`: {name: [f32, ...]}."""
    return {
        "tool": "test-fixture",
        "loss": loss,
        "gradients": {name: {"shape": [len(vals)], "grad": vals, "weight": [0.0] * len(vals)} for name, vals in gradients.items()},
    }


class DerivationTests(unittest.TestCase):
    def test_bf16_unit_roundoff_matches_the_documented_figure(self):
        # u = 2^-8 = 0.00390625 -- half of bf16's machine epsilon 2^-7.
        self.assertAlmostEqual(cgo.BF16_UNIT_ROUNDOFF, 0.00390625, places=10)

    def test_relative_error_bound_scales_as_sqrt_of_layers_times_hidden(self):
        # bound = sqrt(num_layers * hidden_size) * u -- root-sum-square
        # composition across BOTH axes (see the function's own doc for why
        # inter-layer composition uses the same sqrt(N) shape as the
        # intra-layer accumulation, not a linear sum).
        b1 = cgo.derive_relative_error_bound(num_layers=1, hidden_size=1024)
        b2 = cgo.derive_relative_error_bound(num_layers=2, hidden_size=1024)
        self.assertAlmostEqual(b2, math.sqrt(2) * b1, places=9)
        b_h1 = cgo.derive_relative_error_bound(num_layers=1, hidden_size=1024)
        b_h4 = cgo.derive_relative_error_bound(num_layers=1, hidden_size=4096)
        self.assertAlmostEqual(b_h4, 2 * b_h1, places=6)  # sqrt(4096)/sqrt(1024) == 2

    def test_relative_error_bound_rejects_nonpositive_inputs(self):
        with self.assertRaises(ValueError):
            cgo.derive_relative_error_bound(0, 1024)
        with self.assertRaises(ValueError):
            cgo.derive_relative_error_bound(28, 0)

    def test_cosine_floor_is_derived_not_fitted_and_decreases_with_depth(self):
        shallow = cgo.derive_cosine_floor(num_layers=1, hidden_size=32)
        deep = cgo.derive_cosine_floor(num_layers=28, hidden_size=1024)
        self.assertGreater(shallow, deep, "a deeper/wider network must derive a LOOSER floor")
        self.assertLessEqual(shallow, 1.0)
        self.assertGreaterEqual(deep, -1.0)

    def test_cosine_floor_never_raises_on_a_very_deep_config(self):
        # derive_relative_error_bound(1000, 100000) is huge -- cos() must
        # still return a value in [-1, 1], never raise.
        floor = cgo.derive_cosine_floor(num_layers=1000, hidden_size=100_000)
        self.assertGreaterEqual(floor, -1.0)
        self.assertLessEqual(floor, 1.0)


class ComparatorMathTestsMixin:
    """Mixin run under BOTH numpy-available and pure-Python-fallback
    configurations (see the two concrete subclasses below) -- every
    assertion here must hold identically under both paths.
    """

    def test_identical_vectors_have_cosine_one(self):
        self.assertAlmostEqual(cgo.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=9)

    def test_orthogonal_vectors_have_cosine_zero(self):
        self.assertAlmostEqual(cgo.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=9)

    def test_opposite_vectors_have_cosine_minus_one(self):
        self.assertAlmostEqual(cgo.cosine_similarity([1.0, 2.0], [-1.0, -2.0]), -1.0, places=9)

    def test_known_angle_matches_hand_computed_cosine(self):
        # a=(1,0), b=(1,1): cos(theta) = a.b/(|a||b|) = 1/sqrt(2).
        self.assertAlmostEqual(cgo.cosine_similarity([1.0, 0.0], [1.0, 1.0]), 1.0 / math.sqrt(2), places=9)

    def test_zero_vector_is_a_finite_zero_not_nan_or_crash(self):
        """FAMILY F NON-VACUOUS CONTROL: a genuinely all-zero gradient (the
        REAL case `grad_oracle.rs` documents for `lora_a` at a fresh
        `LoraInitMode::ZerosB` init -- confirmed empirically, not
        hypothetical) must not divide-by-zero into `NaN`/`inf`. `NaN >=
        floor` is `False` in Python too, so a naive `cosine >= floor`
        check WOULD correctly reject a `NaN` here -- but only by accident,
        and only if nothing upstream ever treats `NaN` as truthy first
        (e.g. an `if cosine:` check, which `NaN` passes). This test pins
        the finite `0.0`, removing that dependency on downstream luck.
        """
        result = cgo.cosine_similarity([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertEqual(result, 0.0)
        self.assertFalse(math.isnan(result))
        self.assertFalse(math.isinf(result))

    def test_max_abs_delta_and_ratio(self):
        stats = cgo.compare_tensor("t", [1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
        self.assertAlmostEqual(stats["max_abs_delta"], 0.5, places=9)
        self.assertAlmostEqual(stats["max_abs_delta_over_max_signal"], 0.5 / 3.5, places=9)

    def test_compare_tensor_length_mismatch_raises_loudly(self):
        with self.assertRaises(ValueError):
            cgo.compare_tensor("t", [1.0, 2.0], [1.0])

    def test_compare_reports_identical_dumps_passes_with_cosine_one(self):
        report = make_report(0.31, {"layer.0.Wqkv.lora_b": [0.1, -0.2, 0.3, 0.05]})
        result = cgo.compare_reports(report, report, cosine_floor=0.999)
        self.assertEqual(result["overall_cosine_similarity"], 1.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["only_in_a"], [])
        self.assertEqual(result["only_in_b"], [])

    def test_compare_reports_name_mismatch_is_loud_never_silently_skipped(self):
        report_a = make_report(0.3, {"layer.0.Wqkv.lora_b": [0.1, 0.2]})
        report_b = make_report(0.3, {"layer.1.Wqkv.lora_b": [0.1, 0.2]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(result["only_in_a"], ["layer.0.Wqkv.lora_b"])
        self.assertEqual(result["only_in_b"], ["layer.1.Wqkv.lora_b"])
        self.assertFalse(result["passed"], "a name mismatch must fail even if nothing matched is compared")

    def test_compare_reports_gross_defect_fails_the_derived_floor(self):
        """The lead's own named worst case: a gradient magnitude 3x off on
        a shared tensor. This is NOT isotropic bf16 rounding noise -- it
        is a systematic scale error a gradient-DIRECTION oracle catches
        even though every element still points the same SIGN (a naive
        "same sign everywhere" check would miss this; cosine on a
        UNIFORMLY-scaled-but-not-uniformly-so vector still degrades once
        the scale factor varies per element, which is what a real defect
        does -- this fixture models that with a per-element scale drift).
        """
        base = [0.10, -0.20, 0.05, 0.30, -0.15, 0.08, -0.02, 0.22]
        drifted = [v * (3.0 if i % 2 == 0 else 1.0) for i, v in enumerate(base)]
        report_a = make_report(0.3, {"t": base})
        report_b = make_report(0.3, {"t": drifted})
        # (1, 32): the tiny fixture's own shape (1 layer, hidden=32) --
        # small enough for the derived floor to be INFORMATIVE (see
        # test_cosine_floor_is_uninformative_at_full_modernbert_large_depth
        # below for why ModernBERT-large's own 28/1024 derives an
        # uninformative -1.0 floor and must not be used here).
        result = cgo.compare_reports(report_a, report_b, cosine_floor=cgo.derive_cosine_floor(1, 32))
        self.assertLess(result["overall_cosine_similarity"], 0.999)
        self.assertFalse(result["passed"])

    def test_compare_reports_small_isotropic_noise_clears_a_derived_floor(self):
        """The other half of the same claim: genuine small, per-element,
        roughly-isotropic noise (modeling bf16 rounding, not a defect)
        MUST clear a floor derived at a realistic depth/width -- a floor
        so tight it rejects ordinary bf16 noise would be useless (every
        real sweep would false-fail).
        """
        import random

        rng = random.Random(12345)
        base = [rng.uniform(-1.0, 1.0) for _ in range(4096)]
        # (1, 32): see test_compare_reports_gross_defect_fails_the_derived_floor
        # for why this scale (not ModernBERT-large's 28/1024) is the one
        # that derives an INFORMATIVE floor -- at ModernBERT-large's own
        # depth this assertion would be trivially true (floor == -1.0) and
        # would not actually be testing anything.
        eps = cgo.derive_relative_error_bound(num_layers=1, hidden_size=32)
        noisy = [v + rng.uniform(-eps, eps) * abs(v) for v in base]
        report_a = make_report(0.3, {"t": base})
        report_b = make_report(0.3, {"t": noisy})
        floor = cgo.derive_cosine_floor(num_layers=1, hidden_size=32)
        result = cgo.compare_reports(report_a, report_b, cosine_floor=floor)
        self.assertGreaterEqual(result["overall_cosine_similarity"], floor)
        self.assertTrue(result["passed"])

    def test_cosine_floor_is_too_loose_to_catch_a_3x_defect_at_full_modernbert_large_depth(self):
        """HONEST DISCLOSURE, pinned as a test rather than left as only a
        docstring claim: even the root-sum-square (not linear) formula
        derives a floor around -0.40 at ModernBERT-large's real depth
        (`num_layers=28`, `hidden_size=1024`, the CLI's own defaults) with
        the default 3x safety factor -- loose enough that the SAME 3x
        per-element magnitude defect
        `test_compare_reports_gross_defect_fails_the_derived_floor` catches
        at the tiny fixture's shape (1 layer, hidden=32) would NOT fail
        this floor (cosine ~0.87 still clears -0.40). This is not a
        derivation bug: a genuinely statistical bound gets looser with
        depth, and this test exists so that gap is a documented, asserted
        FACT next to the code, not a surprise discovered on a real sweep.
        A caller comparing a REAL ModernBERT-large sweep should scope
        `--num-layers`/`--hidden-size` to the SPECIFIC tensor being checked
        (e.g. the LAST layer's adapter, which only backprops through one
        layer's own rounding, not all 28) rather than trust the whole-model
        default to be tight.
        """
        floor = cgo.derive_cosine_floor(num_layers=28, hidden_size=1024)
        self.assertLess(floor, 0.0)
        self.assertGreater(floor, -1.0, "the improved formula must not saturate at -1.0 the way the naive linear-chaining formula did")

        base = [0.10, -0.20, 0.05, 0.30, -0.15, 0.08, -0.02, 0.22]
        drifted = [v * (3.0 if i % 2 == 0 else 1.0) for i, v in enumerate(base)]
        result = cgo.compare_reports(make_report(0.3, {"t": base}), make_report(0.3, {"t": drifted}), cosine_floor=floor)
        self.assertTrue(result["passed"], "this defect is EXPECTED to clear the full-depth floor -- that is the disclosed gap")


class ComparatorMathTestsWhateverNumpyIsAvailable(ComparatorMathTestsMixin, unittest.TestCase):
    """Runs under this environment's ACTUAL numpy availability (whatever
    that is) -- `cgo.HAVE_NUMPY` at import time.
    """


class ComparatorMathTestsForcedPureFallback(ComparatorMathTestsMixin, unittest.TestCase):
    """Forces `cgo.HAVE_NUMPY = False` for the duration of each test, so
    the pure-Python fallback path is exercised regardless of whether numpy
    happens to be importable in the environment running this suite (it is
    NOT, in the sandbox this round was built in -- see the dispatch
    verdict's mutation-triage note).
    """

    def setUp(self):
        self._orig = cgo.HAVE_NUMPY
        cgo.HAVE_NUMPY = False

    def tearDown(self):
        cgo.HAVE_NUMPY = self._orig


if __name__ == "__main__":
    unittest.main()
