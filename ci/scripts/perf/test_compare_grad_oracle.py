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

import itertools
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_grad_oracle as cgo  # noqa: E402

# Every `make_report()` call gets a UNIQUE default `tool` value (a monotonic
# counter, not a fixed literal) — the class-level fix for "compare a.json
# a.json passes" (advisory (6), this round's audit): TWO SEPARATE calls
# (the overwhelmingly common shape below — one for `report_a`, one for
# `report_b`) now differ by construction, so `_same_producer_violation`'s
# new same-tool check does NOT spuriously fire on the ~30 existing
# gradient-math tests in this file, none of which were EVER testing
# same-producer-ness in the first place. The one test that IS a genuine
# same-producer/self-consistency check
# (`test_compare_reports_identical_dumps_passes_with_cosine_one`, which
# passes the literal SAME dict object twice) opts in explicitly via
# `allow_same_producer=True`.
_MAKE_REPORT_TOOL_COUNTER = itertools.count()


def make_report(loss, gradients, **overrides):
    """`gradients`: {name: [f32, ...]}.

    Fills in a MATCHING, premise-satisfying run-identity/weight-provenance
    envelope by default (`lora_weights_in` set, standard seed/batch/etc.,
    `batch_token_id_sums` present, every `RUN_IDENTITY_FIELDS` entry) — F3's
    fix made `compare_reports` check all of these, so a test that only
    cares about the gradient-DIRECTION math (the vast majority below) must
    not ALSO have to hand-build a full premise or it would spuriously fail
    on a premise violation it never meant to test. Pass e.g.
    `lora_weights_in=None` or `seed=7` via `overrides` to specifically
    construct a premise violation (see `PremiseAndWeightChecks` below).
    """
    report = {
        "tool": f"test-fixture-{next(_MAKE_REPORT_TOOL_COUNTER)}",
        "loss": loss,
        "seed": 1,
        "batch": 2,
        "seq": 4,
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "target_modules": ["Wqkv"],
        "batched_forward": True,
        "backbone_dtype": "f32",
        # Fixture placeholder digests -- both sides get the SAME literal
        # constant via TWO separate `make_report()` calls (never computed
        # from real bytes here; this file's whole point is testing the
        # comparator's ARITHMETIC/premise logic in isolation from either
        # real producer -- see `test_grad_oracle_cross_producer_parity.py`
        # for the version driven against real dumps).
        "checkpoint_config_sha256": "a" * 64,
        "checkpoint_weights_sha256": "b" * 64,
        "checkpoint_weights_size_bytes": 1024,
        "lora_weights_in": "shared_lora.safetensors",
        "batch_token_id_sums": [11, 22, 33],
        "gradients": {name: {"shape": [len(vals)], "grad": vals, "weight": [0.0] * len(vals)} for name, vals in gradients.items()},
    }
    report.update(overrides)
    return report


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

    def test_derive_cosine_floor_is_non_positive_at_modernbert_large_defaults(self):
        """F2 REPRODUCTION, pinned as a numeric fact: `main()`'s own
        DEFAULT arguments (`--num-layers 28 --hidden-size 1024`) derive a
        floor `~-0.4018` — cited exactly by `derive_cosine_floor`'s own
        docstring correction. `eps = sqrt(28*1024) * 2**-8 ~= 0.6614`,
        `3*eps ~= 1.9843` radians, `cos(1.9843) ~= -0.4018`. A floor this
        far below zero is cleared by an angle up to ~113.7 degrees,
        including an EXACT 90-degree rotation (cosine 0.0) — this is the
        numeric fact `main()`'s refusal (see `MainEntryPointRefusalTests`
        below) exists to stop from silently printing PASS.
        """
        floor = cgo.derive_cosine_floor(num_layers=28, hidden_size=1024)
        self.assertAlmostEqual(floor, -0.4018325133266392, places=9)
        self.assertLessEqual(floor, 0.0)
        # And the specific claim: a 90-degree rotation clears it.
        self.assertGreaterEqual(0.0, floor)

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

    def test_dot_and_max_abs_delta_raise_on_length_mismatch_under_whichever_arm_is_active(self):
        """UNGATED companion of `ArmParityTests`' own length-mismatch
        coverage (that class `skipUnless(cgo.HAVE_NUMPY, ...)`s, so it does
        not run at all in an environment without numpy -- see that class's
        own doc for the CI Guard disclosure). This mixin runs under BOTH
        `ComparatorMathTestsWhateverNumpyIsAvailable` and
        `ComparatorMathTestsForcedPureFallback`, so `_require_same_length`
        gets AT LEAST one arm's worth of real coverage regardless of
        whether numpy is importable here.
        """
        with self.assertRaises(ValueError):
            cgo._dot([1.0, 2.0], [1.0])
        with self.assertRaises(ValueError):
            cgo._max_abs_delta([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_compare_reports_identical_dumps_passes_with_cosine_one(self):
        """A genuine same-producer/self-consistency check (the SAME dict
        object on both sides) — opts INTO `allow_same_producer=True`
        explicitly, since this is exactly the case
        `_same_producer_violation` exists to refuse by default (see that
        function's own doc and `SameProducerGuardTests` below).
        """
        report = make_report(0.31, {"layer.0.Wqkv.lora_b": [0.1, -0.2, 0.3, 0.05]})
        result = cgo.compare_reports(report, report, cosine_floor=0.999, allow_same_producer=True)
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


class PremiseAndWeightChecks(unittest.TestCase):
    """F3 REGRESSION (audit finding on PR #372): `compare_reports` used to
    read ONLY `report[...]['gradients'][name]['grad']` — never `weight`,
    never the run-identity fields, never `batch_token_id_sums` — so two
    dumps taken at DIFFERENT weights (or different batches, or different
    configs) could still print `PASS`. Every test here constructs exactly
    ONE premise violation (all else matching) and asserts `passed is
    False` — the REPRODUCTION from the audit (`weight=[0,0,0,0]` vs
    `weight=[9,9,9,9]`, no `--lora-weights-in` recorded, identical
    gradients) is `test_reproduction_zero_vs_nine_weight_with_matching_gradients_is_not_a_pass`
    below, driven at `main()` (the real entry point), not `compare_reports`
    called with hand-built literals in isolation.
    """

    def test_weight_mismatch_fails_even_with_cosine_one(self):
        report_a = make_report(0.3, {"layer.0.Wqkv.lora_b": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"layer.0.Wqkv.lora_b": [1.0, 2.0, 3.0]})
        report_a["gradients"]["layer.0.Wqkv.lora_b"]["weight"] = [0.0, 0.0, 0.0, 0.0]
        report_b["gradients"]["layer.0.Wqkv.lora_b"]["weight"] = [9.0, 9.0, 9.0, 9.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(result["overall_cosine_similarity"], 1.0, "gradients are identical by construction")
        self.assertFalse(result["passed"], "identical gradients at DIFFERENT weights must not pass")
        self.assertTrue(result["weight_mismatches"], "the weight mismatch must be reported, not silently absorbed")

    def test_weight_tolerance_is_ulp_relative_not_a_fixed_absolute_constant(self):
        """Advisory (ii), round-2 audit fix on PR #372: the OLD bound was a
        fixed `1e-4` absolute constant -- loose enough to ALSO pass a
        SMALLER real content mismatch than the audit's own `0 vs 9`
        reproduction. This pins the mechanism directly: a `5e-5` per-element
        delta at magnitude ~1.0 is BELOW the old fixed `1e-4` bound (the old
        code would have silently PASSED this as "identical weights") but is
        ~400x the new f32-ULP-relative bound (`WEIGHT_MATCH_ULPS *
        F32_EPSILON ~= 9.5e-7` at this scale) and must FAIL under the fix.
        The lead's own measured real-world figure, `1.86e-9`, is the OTHER
        half of the same lattice cell below: it must still PASS.
        """
        old_fixed_bound_would_have_passed_this = 5e-5
        self.assertLess(
            old_fixed_bound_would_have_passed_this, 1e-4,
            "fixture assumption: this delta must be SMALLER than the retired fixed bound",
        )
        new_ulp_bound_at_this_scale = cgo.WEIGHT_MATCH_ULPS * cgo.F32_EPSILON  # scale ~1.0
        self.assertGreater(
            old_fixed_bound_would_have_passed_this, new_ulp_bound_at_this_scale,
            "fixture assumption: this delta must be LARGER than the new ULP-relative bound",
        )

        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_a["gradients"]["t"]["weight"] = [1.0, 1.0, 1.0]
        report_b["gradients"]["t"]["weight"] = [1.0 + old_fixed_bound_would_have_passed_this, 1.0, 1.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"], "a real, smaller-than-1e-4 weight delta must now FAIL")
        self.assertTrue(result["weight_mismatches"])

    def test_weight_tolerance_still_passes_the_leads_own_measured_real_run_delta(self):
        """The other half of the same lattice cell: `1.86e-9` (the lead's
        OWN measured `max|w_jammi - w_torch|` over 224 tensors on a real
        A100 run, see `torch_grad_oracle.py`'s PROVENANCE banner) must still
        clear the new, TIGHTER bound -- the fix must not have overshot into
        false-failing the exact real measurement that motivated keeping a
        bound at all.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_a["gradients"]["t"]["weight"] = [1.0, 1.0, 1.0]
        report_b["gradients"]["t"]["weight"] = [1.0 + 1.86e-9, 1.0, 1.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(result["weight_mismatches"], [])
        self.assertTrue(result["passed"])

    def test_missing_lora_weights_in_on_either_side_fails(self):
        matching = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        for missing_side in ("a", "b"):
            report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, lora_weights_in=(None if missing_side == "a" else matching["lora_weights_in"]))
            report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, lora_weights_in=(None if missing_side == "b" else matching["lora_weights_in"]))
            result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
            self.assertFalse(result["passed"], f"missing lora_weights_in on side {missing_side} must refuse")
            self.assertTrue(result["premise_violations"])

    def test_run_identity_mismatch_fails(self):
        for field, other_value in (
            ("seed", 999),
            ("batch", 64),
            ("seq", 512),
            ("lora_rank", 16),
            ("lora_alpha", 8.0),
            ("target_modules", ["Wo"]),
            ("batched_forward", False),
            ("backbone_dtype", "bf16"),
            ("checkpoint_config_sha256", "c" * 64),
            ("checkpoint_weights_sha256", "d" * 64),
            ("checkpoint_weights_size_bytes", 2048),
        ):
            report_a = make_report(0.3, {"t": [1.0, 2.0]})
            report_b = make_report(0.3, {"t": [1.0, 2.0]}, **{field: other_value})
            result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
            self.assertFalse(result["passed"], f"a {field!r} mismatch must refuse")
            self.assertTrue(
                any(field in v for v in result["premise_violations"]),
                f"expected a premise_violations entry naming {field!r}, got {result['premise_violations']!r}",
            )

    def test_run_identity_field_missing_from_both_dumps_fails(self):
        """ITEM 2 REPRODUCTION: `_premise_violations`'s per-field loop used
        to `report.get(field)` on BOTH sides, then compare — a field absent
        from BOTH dumps compared `None == None` and silently PASSED. This is
        the class-level closure, driven over EVERY `RUN_IDENTITY_FIELDS`
        entry (not only `batch_token_id_sums`, which already had a
        correctly-`or`-gated presence check before this round).
        """
        for field in cgo.RUN_IDENTITY_FIELDS:
            report_a = make_report(0.3, {"t": [1.0, 2.0]})
            report_b = make_report(0.3, {"t": [1.0, 2.0]})
            del report_a[field]
            del report_b[field]
            result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
            self.assertFalse(
                result["passed"],
                f"{field!r} missing from BOTH dumps must not silently compare None == None and pass",
            )
            self.assertTrue(
                any(field in v for v in result["premise_violations"]),
                f"expected a premise_violations entry naming {field!r}, got {result['premise_violations']!r}",
            )

    def test_run_identity_field_missing_from_one_dump_fails(self):
        """The one-sided companion of the test above -- was ALREADY caught
        correctly before this round (`None != <value>`), pinned here per
        field so the whole `RUN_IDENTITY_FIELDS` tuple has both halves of
        the presence lattice covered, not just the both-missing half.
        """
        for field in cgo.RUN_IDENTITY_FIELDS:
            report_a = make_report(0.3, {"t": [1.0, 2.0]})
            report_b = make_report(0.3, {"t": [1.0, 2.0]})
            del report_a[field]
            result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
            self.assertFalse(result["passed"], f"{field!r} missing from ONE dump must refuse")
            self.assertTrue(any(field in v for v in result["premise_violations"]))

    def test_batch_token_id_sums_mismatch_fails(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0]}, batch_token_id_sums=[1, 2, 3])
        report_b = make_report(0.3, {"t": [1.0, 2.0]}, batch_token_id_sums=[1, 2, 4])
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(any("batch_token_id_sums" in v for v in result["premise_violations"]))

    def test_batch_token_id_sums_missing_fails_not_silently_skipped(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0]})
        del report_a["batch_token_id_sums"]
        report_b = make_report(0.3, {"t": [1.0, 2.0]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(any("batch_token_id_sums" in v for v in result["premise_violations"]))

    def test_matching_premise_and_matching_weight_passes(self):
        """Positive control: the premise/weight checks above must not
        false-fail a genuinely matching pair — otherwise every test in
        `ComparatorMathTestsMixin` above (which now also exercises this
        code path via the updated `make_report` default) would be a false
        negative waiting to happen.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertTrue(result["passed"])
        self.assertEqual(result["weight_mismatches"], [])
        self.assertEqual(result["premise_violations"], [])

    def test_weight_nan_on_one_side_fails_never_reads_as_max_abs_delta_zero(self):
        """ITEM 3 REPRODUCTION (the auditor's own reproduction, reproduced
        here as a RED->GREEN test): a NaN weight element on ONE side used to
        make `delta > tol` compare `False` (NaN comparisons are always
        `False` in IEEE-754 ordering), so `_weight_max_violation` counted it
        as "not bad" and `max_delta` could silently read `0.0` via a naive
        `max(0.0, nan)` reduction — PASS, exit 0, max_abs_delta=0.0. Must now
        FAIL, and the weight_mismatches entry must name the nonfinite cause
        explicitly, not report a numeric max_abs_delta at all.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_a["gradients"]["t"]["weight"] = [1.0, 1.0, 1.0]
        report_b["gradients"]["t"]["weight"] = [1.0, float("nan"), 1.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(result["weight_mismatches"])
        self.assertTrue(
            any("NaN" in m or "nonfinite" in m for m in result["weight_mismatches"]),
            f"expected the nonfinite cause to be named, got {result['weight_mismatches']!r}",
        )

    def test_weight_nan_on_both_sides_still_fails(self):
        """A NaN on BOTH sides at the SAME index is not 'equal enough'
        either — a nonfinite weight can never certify 'the identical weight
        file' regardless of which side(s) carry it.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_a["gradients"]["t"]["weight"] = [1.0, float("nan"), 1.0]
        report_b["gradients"]["t"]["weight"] = [1.0, float("nan"), 1.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(result["weight_mismatches"])

    def test_weight_infinity_fails_the_same_way_as_nan(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_a["gradients"]["t"]["weight"] = [1.0, 1.0, 1.0]
        report_b["gradients"]["t"]["weight"] = [1.0, float("inf"), 1.0]
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(result["weight_mismatches"])

    def test_weight_max_violation_reports_nan_max_delta_never_a_silent_zero(self):
        """Drives `_weight_max_violation` directly, under BOTH arms (numpy
        forced on, pure-Python forced off) — `bad` must count the nonfinite
        element and `max_delta` must be NaN (not `0.0`), in BOTH arms,
        mirroring the arm-parity discipline `ArmParityTests` pins for the
        gradient-side helpers.
        """
        wa = [1.0, 1.0, 1.0]
        wb = [1.0, float("nan"), 1.0]
        for forced_numpy in (True, False):
            if forced_numpy and not cgo.HAVE_NUMPY:
                continue  # this arm genuinely does not exist here -- see ArmParityTests' skip
            orig = cgo.HAVE_NUMPY
            cgo.HAVE_NUMPY = forced_numpy
            try:
                bad, max_delta = cgo._weight_max_violation(wa, wb)
            finally:
                cgo.HAVE_NUMPY = orig
            self.assertGreaterEqual(bad, 1, f"forced_numpy={forced_numpy}: nan element must count as bad")
            self.assertTrue(math.isnan(max_delta), f"forced_numpy={forced_numpy}: max_delta must be NaN, got {max_delta}")


class BackboneDtypeSpellingNormalizationTests(unittest.TestCase):
    """B1 REGRESSION (audit finding on PR #372, round 2): jammi's
    `grad_oracle.rs` has always emitted `backbone_dtype: "f32"`;
    `torch_grad_oracle.py`, before this fix, emitted torch's bare CLI-flag
    spelling `"fp32"` straight through — so EVERY jammi-f32-vs-torch-f32
    comparison (including this oracle's own near-perfect control) was
    UNRUNNABLE, refused on a spurious SPELLING mismatch despite both sides
    having run at the identical, actual precision. See
    `test_grad_oracle_cross_producer_parity.py` (a SEPARATE file, requires a
    real cargo build + a real torch venv) for the version of this same
    finding driven against the REAL emitters, not the fixture dicts below.
    """

    def test_normalize_backbone_dtype_maps_the_one_known_legacy_spelling(self):
        self.assertEqual(cgo.normalize_backbone_dtype("fp32"), "f32")

    def test_normalize_backbone_dtype_leaves_canonical_and_unknown_values_alone(self):
        self.assertEqual(cgo.normalize_backbone_dtype("f32"), "f32")
        self.assertEqual(cgo.normalize_backbone_dtype("bf16"), "bf16")
        self.assertEqual(cgo.normalize_backbone_dtype("f16"), "f16")
        self.assertEqual(cgo.normalize_backbone_dtype("xyzzy"), "xyzzy")
        self.assertIsNone(cgo.normalize_backbone_dtype(None))

    def test_f32_vs_legacy_fp32_spelling_is_not_a_premise_violation(self):
        """THE B1 REPRODUCTION, driven at `compare_reports`: jammi's `f32`
        against a LEGACY torch dump's `fp32` (same precision, different
        spelling) must NOT be flagged — this is the exact scenario that made
        this oracle's own near-perfect f32-vs-f32 control unrunnable.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, backbone_dtype="f32")
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, backbone_dtype="fp32")
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertTrue(result["passed"], f"premise_violations={result['premise_violations']!r}")
        self.assertEqual(result["premise_violations"], [])

    def test_bf16_vs_f16_is_still_a_real_premise_violation_not_swallowed_by_normalization(self):
        """NEGATIVE CONTROL: the normalization map narrows exactly ONE
        known legacy SPELLING of the SAME precision together — it must
        never widen into "any dtype string is close enough". Two genuinely
        DIFFERENT precisions must still refuse.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, backbone_dtype="bf16")
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, backbone_dtype="f16")
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(any("backbone_dtype" in v for v in result["premise_violations"]))


class RunIdentityFieldCanonicalizationLattice(unittest.TestCase):
    """CLASS-LEVEL closure of the round-2 `backbone_dtype`-only special case
    (the old `if field == "backbone_dtype":` branch in `_premise_violations`,
    whose own inline comment said every OTHER `RUN_IDENTITY_FIELDS` entry
    "is compared as-is"). This class drives `compare_reports` (never a
    helper called with bespoke literals) once PER FIELD in
    `cgo.RUN_IDENTITY_FIELDS`, with two cells per field where a
    representational gap between the two INDEPENDENT producers
    (`crates/jammi-bench/src/grad_oracle.rs`,
    `crates/jammi-bench/reference/torch_grad_oracle.py`) is actually
    representable:

    - REPRESENTATIONALLY-DIFFERENT-BUT-SEMANTICALLY-EQUAL -> must NOT be a
      premise violation.
    - GENUINELY DIFFERENT -> must still be a premise violation (the
      canonicalization must narrow ONLY the representational gap, never
      widen into "any value of this field is close enough" -- the same
      non-widening discipline `BackboneDtypeSpellingNormalizationTests`
      already pins for `backbone_dtype` alone).

    `target_modules`'s reordered-but-same-set cell
    (`test_target_modules_reordered_same_set_is_not_a_violation`) is the
    LIVE INSTANCE this round's dispatch reproduced directly (RED before
    `normalize_target_modules` is wired into `_premise_violations` via
    `canonicalize_identity_field`, GREEN after -- see that function's own
    doc for why a per-field DISPATCH TABLE, not a second inline
    `if field == ...` special case, is what actually closes the class
    rather than moving the point fix one field over).

    `seed`/`batch`/`seq`/`lora_rank`/`batched_forward` have NO known
    representational gap to canonicalize away: `grad_oracle.rs`'s
    `GradOracleReport` fields (`seed: u64`, `batch: usize`, `seq: usize`,
    `lora_rank: usize`, `batched_forward: bool`) serialize through
    `#[derive(Serialize)]`/`serde_json` to a bare JSON number/boolean;
    `torch_grad_oracle.py`'s `argparse` declarations for the same CLI flags
    (`type=int` for the four numeric fields, `action=
    argparse.BooleanOptionalAction` for `--batched-forward`) produce a
    native Python `int`/`bool` fed straight into `json.dump` -- Python's
    `json` module serializes an `int` to a bare JSON number and a `bool` to
    `true`/`false`, never a string or a float. So there is no cross-producer
    spelling/type gap for these five fields to close (unlike
    `backbone_dtype`, whose CLI-flag spelling genuinely differs between the
    two stacks' argument vocabularies, or `target_modules`, whose ORDER is
    genuinely uncontrolled). This class still tests these five fields two
    ways: (1) a MATCHING pair passes (the positive control that this
    round's changes have not accidentally penalized a correctly-configured
    comparison), and (2) a fabricated cross-TYPE variant (e.g. a
    stringified `seed`) -- which NEITHER real producer ever emits, but
    which this comparator must still correctly REJECT as a genuine premise
    violation rather than silently coerce; registering a canonicalizer for
    a gap that does not exist in either real producer would only widen what
    counts as a match with no corresponding representational need.
    """

    def _result(self, **overrides_b):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, **overrides_b)
        return cgo.compare_reports(report_a, report_b, cosine_floor=0.9)

    # ---- target_modules ---------------------------------------------------

    def test_target_modules_reordered_same_set_is_not_a_violation(self):
        """THE REPRODUCTION this round closes: two dumps whose
        `target_modules` name the identical SET (`{"Wqkv", "Wo", "Wi"}`) in
        a different order must not be flagged -- order is not a
        semantically meaningful part of jammi's OWN consumer of this list
        (`jammi_lora::config::should_apply_lora`'s `.any(|t| module_name ==
        t || module_name.ends_with(t))` in `crates/jammi-lora/src/config.rs`
        is an UNORDERED existence check over the whole slice), and both
        producers build this field by literally splitting the operator's
        `--target-modules` CLI string on commas in whatever order the
        operator typed it (`main.rs`'s `target_modules.split(',')...`
        collect, `torch_grad_oracle.py`'s `args.target_modules.split(",")`
        list comprehension).
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, target_modules=["Wqkv", "Wo", "Wi"])
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, target_modules=["Wo", "Wqkv", "Wi"])
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertTrue(result["passed"], f"premise_violations={result['premise_violations']!r}")
        self.assertEqual(result["premise_violations"], [])

    def test_target_modules_different_set_is_still_a_violation(self):
        """NEGATIVE CONTROL: a genuinely different SET (not just a
        different order of the same set) must still refuse -- the
        canonicalizer narrows only ORDER, never MEMBERSHIP.
        """
        result = self._result(target_modules=["Wo", "Wi"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("target_modules" in v for v in result["premise_violations"]))

    def test_target_modules_duplicate_entry_is_still_a_violation(self):
        """NEGATIVE CONTROL, narrower than a different set: `["Wqkv",
        "Wqkv"]` and `["Wqkv"]` name the SAME set but a DIFFERENT multiset
        (a duplicate present on one side only) -- `normalize_target_modules`
        sorts without deduplicating specifically so this case is still
        caught (see that function's own doc for why sortedness, not
        set-collapsing, is the chosen narrowing).
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, target_modules=["Wqkv", "Wqkv"])
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, target_modules=["Wqkv"])
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(any("target_modules" in v for v in result["premise_violations"]))

    def test_normalize_target_modules_narrows_order_only(self):
        self.assertEqual(
            cgo.normalize_target_modules(["Wo", "Wqkv", "Wi"]),
            cgo.normalize_target_modules(["Wqkv", "Wo", "Wi"]),
        )
        self.assertNotEqual(
            cgo.normalize_target_modules(["Wqkv", "Wqkv"]),
            cgo.normalize_target_modules(["Wqkv"]),
        )
        self.assertIsNone(cgo.normalize_target_modules(None))

    # ---- backbone_dtype (already covered above; pinned here too so this
    # class is a genuinely complete per-field lattice on its own) ----------

    def test_backbone_dtype_legacy_spelling_is_not_a_violation(self):
        result = self._result(backbone_dtype="fp32")  # report_a default is "f32"
        self.assertTrue(result["passed"], f"premise_violations={result['premise_violations']!r}")

    def test_backbone_dtype_genuinely_different_precision_is_still_a_violation(self):
        result = self._result(backbone_dtype="bf16")
        self.assertFalse(result["passed"])
        self.assertTrue(any("backbone_dtype" in v for v in result["premise_violations"]))

    # ---- seed / batch / seq / lora_rank / batched_forward ------------------
    # No known cross-producer representational gap (see this class's own
    # doc) -- these five fields still get a matching-pair positive control
    # and a fabricated cross-type negative control each, so an audit can
    # verify "safe as-is" by RUNNING the claim, not by reading a comment.

    def test_matching_int_and_bool_identity_fields_are_not_violations(self):
        result = self._result()  # no overrides -- everything matches by construction
        self.assertTrue(result["passed"])
        self.assertEqual(result["premise_violations"], [])

    def test_stringified_numeric_identity_fields_are_still_violations(self):
        """NEITHER real producer ever emits a stringified `seed`/`batch`/
        `seq`/`lora_rank` (`grad_oracle.rs`'s fields are native `u64`/
        `usize`; `torch_grad_oracle.py`'s argparse declarations are all
        `type=int`) -- this pins that if one somehow did (a schema
        regression on either side), this comparator still catches it as a
        genuine mismatch rather than silently coercing `"1" == 1` (which
        Python's own `==` already returns `False` for, with NO
        canonicalizer needed -- this test exists to keep it that way on
        purpose, not to add one).
        """
        for field in ("seed", "batch", "seq", "lora_rank"):
            with self.subTest(field=field):
                base = make_report(0.3, {"t": [1.0, 2.0, 3.0]})
                stringified = str(base[field])
                result = self._result(**{field: stringified})
                self.assertFalse(result["passed"], f"{field} type mismatch must still refuse")
                self.assertTrue(any(field in v for v in result["premise_violations"]))

    def test_batched_forward_genuinely_different_value_is_still_a_violation(self):
        result = self._result(batched_forward=False)
        self.assertFalse(result["passed"])
        self.assertTrue(any("batched_forward" in v for v in result["premise_violations"]))

    # ---- lora_alpha (promoted from advisory-only to identity this round) --

    def test_lora_alpha_matching_is_not_a_violation(self):
        result = self._result()  # no overrides -- matches by construction
        self.assertNotIn(
            True, [ "lora_alpha" in v for v in result["premise_violations"] ]
        )

    def test_lora_alpha_genuinely_different_value_is_still_a_violation(self):
        """Advisory (6) closure: `lora_alpha` used to be EXCLUDED from
        `RUN_IDENTITY_FIELDS` on the theory that `max_abs_delta_over_max_signal`
        "already surfaces" a mismatch -- that field never gates `passed` (see
        `compare_tensor`'s own doc), so it gated NOTHING. Now identity.
        """
        result = self._result(lora_alpha=8.0)
        self.assertFalse(result["passed"])
        self.assertTrue(any("lora_alpha" in v for v in result["premise_violations"]))

    # ---- checkpoint content identity (replaces the un-comparable model_dir path) --

    def test_checkpoint_hash_fields_matching_is_not_a_violation(self):
        result = self._result()
        self.assertEqual(result["premise_violations"], [])

    def test_checkpoint_config_sha256_mismatch_is_a_violation(self):
        result = self._result(checkpoint_config_sha256="c" * 64)
        self.assertFalse(result["passed"])
        self.assertTrue(any("checkpoint_config_sha256" in v for v in result["premise_violations"]))

    def test_checkpoint_weights_sha256_mismatch_is_a_violation(self):
        result = self._result(checkpoint_weights_sha256="d" * 64)
        self.assertFalse(result["passed"])
        self.assertTrue(any("checkpoint_weights_sha256" in v for v in result["premise_violations"]))

    def test_checkpoint_weights_size_bytes_mismatch_is_a_violation(self):
        result = self._result(checkpoint_weights_size_bytes=2048)
        self.assertFalse(result["passed"])
        self.assertTrue(any("checkpoint_weights_size_bytes" in v for v in result["premise_violations"]))

    # ---- completeness meta-test (item 5 of this round's audit) ------------

    def test_every_run_identity_field_has_a_lattice_cell(self):
        """CLASS-LEVEL closure of the round-2 hard-coded-field-name gap
        (this class previously named fields directly in its own methods,
        disconnected from `cgo.RUN_IDENTITY_FIELDS`): this test iterates
        `cgo.RUN_IDENTITY_FIELDS` (the single source of truth — see that
        tuple's own doc) and fails LOUDLY if a field has been added there
        without a corresponding entry in `_RUN_IDENTITY_FIELD_LATTICE_COVERAGE`
        below — a NEW identity field that ships with zero lattice coverage
        is exactly the shape of gap this round's audit named (the round-2
        fix's `backbone_dtype`-only special case looked complete but
        silently had none for every other field).
        """
        missing = [f for f in cgo.RUN_IDENTITY_FIELDS if f not in _RUN_IDENTITY_FIELD_LATTICE_COVERAGE]
        self.assertEqual(
            missing, [],
            f"RUN_IDENTITY_FIELDS field(s) {missing!r} have NO lattice test coverage in this "
            "class -- add a per-field test method above (a representationally-different-but-"
            "equal cell if the field has one, plus a genuinely-different negative control "
            "either way) and register the field name in "
            "_RUN_IDENTITY_FIELD_LATTICE_COVERAGE",
        )


# Manually-maintained registry `test_every_run_identity_field_has_a_lattice_cell`
# checks `cgo.RUN_IDENTITY_FIELDS` against — see that test's own doc. Updating
# this set alone does NOT substitute for writing the real per-field test
# methods above; it is the mechanical tripwire that makes forgetting to do so
# loud instead of silent.
_RUN_IDENTITY_FIELD_LATTICE_COVERAGE = frozenset({
    "seed",
    "batch",
    "seq",
    "lora_rank",
    "lora_alpha",
    "target_modules",
    "batched_forward",
    "backbone_dtype",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
})


class PerTensorGatingLattice(unittest.TestCase):
    """B2 REGRESSION (audit finding on PR #372, round 2): `passed` used to
    consult ONLY the overall concatenated cosine — measured, jammi could
    zero 55 of 112 real `lora_b` tensors and still PASS at floor 0.7
    (overall 0.994, diluted by the other 57 agreeing tensors). One test per
    lattice cell named in the audit's own dispatch; every cell drives
    `tensor_clears_floor` AND, where noted, the full `compare_reports`
    entry point — never only the low-level helper in isolation for the
    cells that are the actual audit-named reproduction.
    """

    FLOOR = 0.7

    def test_zero_zero_is_vacuous_and_clears(self):
        stats = cgo.compare_tensor("t", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertTrue(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_zero_nonzero_is_one_sided_and_fails(self):
        stats = cgo.compare_tensor("t", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_nonzero_zero_is_one_sided_and_fails(self):
        stats = cgo.compare_tensor("t", [1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_nan_vs_finite_fails(self):
        stats = cgo.compare_tensor("t", [float("nan"), 1.0], [1.0, 1.0])
        self.assertTrue(stats["has_nonfinite"])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_finite_vs_nan_fails(self):
        stats = cgo.compare_tensor("t", [1.0, 1.0], [float("nan"), 1.0])
        self.assertTrue(stats["has_nonfinite"])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_inf_fails(self):
        stats = cgo.compare_tensor("t", [float("inf"), 1.0], [1.0, 1.0])
        self.assertTrue(stats["has_nonfinite"])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_orthogonal_nonzero_below_floor_fails(self):
        stats = cgo.compare_tensor("t", [1.0, 0.0], [0.0, 1.0])
        self.assertFalse(stats["vacuous"])
        self.assertFalse(stats["one_sided_zero"])
        self.assertFalse(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_nonzero_nonzero_agreeing_clears(self):
        stats = cgo.compare_tensor("t", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertTrue(cgo.tensor_clears_floor(stats, self.FLOOR))

    def test_all_matched_tensors_vacuous_is_not_a_pass(self):
        """Every matched tensor vacuous (no real signal anywhere) must NOT
        pass — `overall_cosine_similarity` is exactly `0.0` in this case
        (an all-zero concatenated vector), which can never clear a floor
        that must itself be `> 0.0` (this comparator's own domain, see
        `floor_domain_violation`), so this already fails via the overall
        check alone -- pinned here explicitly per the audit's own named
        lattice cell, not left as an accidental consequence.
        """
        report_a = make_report(0.3, {"a": [0.0, 0.0], "b": [0.0, 0.0, 0.0]})
        report_b = make_report(0.3, {"a": [0.0, 0.0], "b": [0.0, 0.0, 0.0]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.5)
        self.assertEqual(result["vacuous_tensor_count"], 2)
        self.assertEqual(result["overall_cosine_similarity"], 0.0)
        self.assertFalse(result["passed"], "all-vacuous must never read as a PASS")

    def test_one_of_many_zeroed_tensors_fails_the_whole_comparison(self):
        """THE B2 REPRODUCTION ITSELF, driven at `compare_reports`: models
        the audit's own measured shape (proportionally: many agreeing
        tensors, one zeroed on jammi's side only) at a floor the OLD
        overall-only gate would have passed. Every agreeing tensor is
        EXACTLY identical (cosine 1.0) so the overall concatenated cosine
        stays high even with one tensor zeroed on one side — the per-tensor
        gate is the ONLY thing that can catch this.
        """
        import random

        rng = random.Random(7)
        grads_a, grads_b = {}, {}
        for i in range(20):
            vec = [rng.uniform(-1.0, 1.0) for _ in range(8)]
            grads_a[f"layer.{i}.lora_b"] = vec
            grads_b[f"layer.{i}.lora_b"] = list(vec)  # identical -- cosine 1.0
        # ONE tensor: jammi's side collapsed to zero, torch's side did not.
        grads_a["layer.0.lora_b"] = [0.0] * 8
        grads_b["layer.0.lora_b"] = [0.3, -0.1, 0.2, 0.4, -0.2, 0.1, 0.05, -0.3]

        report_a = make_report(0.3, grads_a)
        report_b = make_report(0.3, grads_b)
        result = cgo.compare_reports(report_a, report_b, cosine_floor=self.FLOOR)

        self.assertGreaterEqual(
            result["overall_cosine_similarity"], self.FLOOR,
            "fixture must reproduce the audit's own shape: the OLD overall-only gate would PASS here",
        )
        self.assertIn("layer.0.lora_b", result["failing_tensor_names"])
        self.assertFalse(result["passed"], "one zeroed tensor among many agreeing ones must still FAIL overall")

    def test_all_tensors_clear_is_a_genuine_pass(self):
        """Positive control: the per-tensor gate must not turn every
        comparison into a FAIL — a genuinely matching pair must still PASS.
        """
        report_a = make_report(0.3, {"a": [1.0, 2.0], "b": [0.5, -0.5, 0.25]})
        report_b = make_report(0.3, {"a": [1.0, 2.0], "b": [0.5, -0.5, 0.25]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(result["failing_tensor_names"], [])
        self.assertTrue(result["passed"])

    def test_worst_tensor_names_surfaces_the_failing_tensor_first(self):
        report_a = make_report(0.3, {"good": [1.0, 2.0, 3.0], "bad": [0.0, 0.0]})
        report_b = make_report(0.3, {"good": [1.0, 2.0, 3.0], "bad": [1.0, 1.0]})
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertIn("bad", result["worst_tensor_names"])
        self.assertEqual(result["worst_tensor_names"][0], "bad", "one-sided-zero must sort before an agreeing tensor")


class MainEntryPointRefusalTests(unittest.TestCase):
    """Drives `cgo.main()` — the REAL entry point (implementer-acceptance
    clause 8), never `derive_cosine_floor`/`compare_reports` called
    directly — for F2's refusal behaviour and F3's premise reproduction.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, report):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as fh:
            json.dump(report, fh)
        return path

    def _run_main(self, argv):
        """Runs `cgo.main(argv)` in-process, capturing stdout/stderr and
        the return code -- never a subprocess, so this exercises the exact
        module state this test file already imported (numpy patched or
        not) rather than a fresh interpreter.
        """
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cgo.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_default_num_layers_hidden_size_refuses_on_an_exactly_orthogonal_pair(self):
        """THE F2 REPRODUCTION: at `--num-layers 28 --hidden-size 1024`
        (ModernBERT-large, this script's own argparse DEFAULTS -- no flags
        passed here beyond the two file paths), two gradient vectors that
        are EXACTLY ORTHOGONAL (cosine 0.0, `[1,2,3,4]` vs `[4,-3,2,-1]` --
        the audit's own reproduction pair: `1*4+2*-3+3*2+4*-1 = 4-6+6-4 =
        0`) must NOT print `PASS`, must NOT return exit 0, and must print
        no line that is exactly `PASS`. Before this fix, `main()` printed
        `PASS` here because the derived floor at these defaults is
        `~-0.402` (see `DerivationTests` above) and `0.0 >= -0.402`.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        report_b = make_report(0.3, {"t": [4.0, -3.0, 2.0, -1.0]})
        path_a = self._write("a.json", report_a)
        path_b = self._write("b.json", report_b)

        code, out, err = self._run_main([path_a, path_b])

        self.assertEqual(code, cgo.EXIT_REFUSED, f"expected a REFUSED exit; stdout={out!r} stderr={err!r}")
        self.assertNotIn("PASS", out.splitlines(), "must never print a bare PASS line when refusing")
        self.assertIn("REFUSED", err)

    def test_explicit_nonpositive_cosine_floor_refuses(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        report_b = make_report(0.3, {"t": [4.0, -3.0, 2.0, -1.0]})
        path_a = self._write("a.json", report_a)
        path_b = self._write("b.json", report_b)

        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "0.0"])

        self.assertEqual(code, cgo.EXIT_REFUSED)
        self.assertNotIn("PASS", out.splitlines())
        self.assertIn("REFUSED", err)
        self.assertIn("explicit", err, "the refusal message must name that the floor came from --cosine-floor")

    def test_explicit_sane_floor_still_catches_the_orthogonal_case(self):
        """The other half of clause 2's requirement: an explicit, positive,
        sane floor must still REFUSE-to-report-PASS on a real (here,
        synthetic-orthogonal) defect -- the fix must not have made the
        comparator unconditionally refuse; it must still be able to FAIL.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        report_b = make_report(0.3, {"t": [4.0, -3.0, 2.0, -1.0]})
        path_a = self._write("a.json", report_a)
        path_b = self._write("b.json", report_b)

        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "0.9"])

        self.assertEqual(code, cgo.EXIT_FAIL)
        self.assertIn("FAIL", out.splitlines())
        self.assertNotIn("PASS", out.splitlines())

    def test_explicit_sane_floor_passes_a_genuinely_matching_pair(self):
        """Positive control for the refusal machinery: a fully
        premise-matching, gradient-identical pair at a sane explicit floor
        must still print PASS and exit 0 -- the fix must not have made
        EVERY invocation refuse or fail.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        path_a = self._write("a.json", report_a)
        path_b = self._write("b.json", report_b)

        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "0.9"])

        self.assertEqual(code, cgo.EXIT_PASS, f"stdout={out!r} stderr={err!r}")
        self.assertIn("PASS", out.splitlines())

    def test_reproduction_zero_vs_nine_weight_with_matching_gradients_is_not_a_pass(self):
        """THE F3 REPRODUCTION, driven at `main()`: two dumps whose
        gradients are IDENTICAL but whose `weight` arrays are `[0,0,0,0]`
        vs `[9,9,9,9]` (the audit's own reproduction values), and neither
        records a loaded `--lora-weights-in` file -- exactly the "omit
        --lora-weights-in on the torch side" scenario the finding names.
        Must not print PASS.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]}, lora_weights_in=None)
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]}, lora_weights_in=None)
        report_a["gradients"]["t"]["weight"] = [0.0, 0.0, 0.0, 0.0]
        report_b["gradients"]["t"]["weight"] = [9.0, 9.0, 9.0, 9.0]
        path_a = self._write("a.json", report_a)
        path_b = self._write("b.json", report_b)

        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "0.9"])

        self.assertEqual(code, cgo.EXIT_FAIL)
        self.assertNotIn("PASS", out.splitlines())
        self.assertIn("PREMISE VIOLATION", err)
        self.assertIn("WEIGHT MISMATCH", err)


class CosineFloorDomainLattice(unittest.TestCase):
    """B3 REGRESSION (audit finding on PR #372, round 2): the previous
    round's refusal only checked `cosine_floor <= 0.0` — `--cosine-floor
    1e-30` PASSED a 90-degree defect (a floor that low is barely above
    zero, still uninformative in practice, but this repo's own domain is
    `(0.0, 1.0]` so it is technically admissible and NOT itself a bug this
    lattice re-litigates); `--cosine-floor nan` was NOT refused (`nan <=
    0.0` is `False`); `--cosine-floor 1.5` was accepted (impossible to ever
    clear). One test per domain edge named in the audit's own dispatch,
    every one driving `main()` — the real entry point — never
    `floor_domain_violation` asserted on in isolation only.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, report):
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as fh:
            json.dump(report, fh)
        return path

    def _run_main(self, argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cgo.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _matching_pair_paths(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0, 4.0]})
        return self._write("a.json", report_a), self._write("b.json", report_b)

    def test_floor_domain_violation_lattice(self):
        # (floor, expect_violation) -- the direct unit-level lattice, before
        # driving main() for the same cells below.
        cases = [
            (float("nan"), True),
            (float("inf"), True),
            (float("-inf"), True),
            (-0.0, True),
            (0.0, True),
            (1e-30, False),
            (0.5, False),
            (1.0, False),
            (1.0000001, True),
            (1.5, True),
        ]
        for floor, expect_violation in cases:
            got = cgo.floor_domain_violation(floor)
            if expect_violation:
                self.assertIsNotNone(got, f"floor={floor!r} should be a domain violation")
            else:
                self.assertIsNone(got, f"floor={floor!r} should be IN the valid domain, got {got!r}")

    def test_nan_floor_refuses_at_main(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "nan"])
        self.assertEqual(code, cgo.EXIT_REFUSED, f"stdout={out!r} stderr={err!r}")
        self.assertNotIn("PASS", out.splitlines())
        self.assertIn("REFUSED", err)

    def test_positive_infinity_floor_refuses_at_main(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "inf"])
        self.assertEqual(code, cgo.EXIT_REFUSED)
        self.assertNotIn("PASS", out.splitlines())

    def test_negative_zero_floor_refuses_at_main(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "-0.0"])
        self.assertEqual(code, cgo.EXIT_REFUSED)
        self.assertNotIn("PASS", out.splitlines())

    def test_tiny_positive_floor_is_in_domain_and_a_genuine_match_passes(self):
        """`1e-30` is technically IN the `(0.0, 1.0]` domain (uninformative
        in practice, but this comparator's domain check is about whether a
        floor CAN discriminate at all, not about picking a good one for the
        caller — `derive_cosine_floor`'s own doc is where the "prefer a
        named, empirically-set floor" guidance lives). Must not refuse.
        """
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "1e-30"])
        self.assertEqual(code, cgo.EXIT_PASS, f"stdout={out!r} stderr={err!r}")
        self.assertIn("PASS", out.splitlines())

    def test_exactly_one_floor_is_in_domain_and_a_perfect_match_passes(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "1.0"])
        self.assertEqual(code, cgo.EXIT_PASS, f"stdout={out!r} stderr={err!r}")

    def test_just_above_one_floor_refuses_at_main(self):
        """THE B3 REPRODUCTION: `--cosine-floor 1.0000001` slipped through
        the OLD `<= 0.0`-only check un-refused — nothing could ever clear
        it (cosine similarity cannot exceed 1.0), so it would have silently
        FAILED every comparison forever, never printing PASS again.
        """
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "1.0000001"])
        self.assertEqual(code, cgo.EXIT_REFUSED, f"stdout={out!r} stderr={err!r}")
        self.assertNotIn("PASS", out.splitlines())

    def test_one_point_five_floor_refuses_at_main(self):
        """THE B3 AUDIT'S OWN NAMED REPRODUCTION: `--cosine-floor 1.5`."""
        path_a, path_b = self._matching_pair_paths()
        code, out, err = self._run_main([path_a, path_b, "--cosine-floor", "1.5"])
        self.assertEqual(code, cgo.EXIT_REFUSED, f"stdout={out!r} stderr={err!r}")
        self.assertNotIn("PASS", out.splitlines())

    def test_empirical_band_is_always_printed_even_on_refusal(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, _err = self._run_main([path_a, path_b, "--cosine-floor", "1.5"])
        self.assertEqual(code, cgo.EXIT_REFUSED)
        self.assertIn("empirical band", out)

    def test_empirical_band_is_printed_on_a_normal_pass(self):
        path_a, path_b = self._matching_pair_paths()
        code, out, _err = self._run_main([path_a, path_b, "--cosine-floor", "0.9"])
        self.assertEqual(code, cgo.EXIT_PASS)
        self.assertIn("empirical band", out)


class VacuousTensorClassificationTests(unittest.TestCase):
    """Lead's live-pod course correction on this PR round (ModernBERT-large,
    A100, tip e62c8a8): at a fresh `LoraInitMode::ZerosB` init, `dL/dA` is
    EXACTLY `0.0` on BOTH stacks for every `lora_a` tensor -- 112 of the
    224 matched tensors in that run. A bare `cosine_similarity() == 0.0`
    for those tensors does not distinguish "these stacks disagree" from
    "neither side has a signal here at all". `is_vacuous_pair`/
    `compare_tensor`'s `vacuous` field/`compare_reports`'s
    `vacuous_tensor_count` close that gap.
    """

    def test_both_sides_exactly_zero_is_vacuous(self):
        self.assertTrue(cgo.is_vacuous_pair([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))

    def test_one_side_zero_other_nonzero_is_not_vacuous(self):
        """The critical negative case: a REAL divergence where one stack's
        gradient collapsed to zero and the other's did not (e.g. a dead
        backward path on one side) must NOT be classified as vacuous --
        vacuous means BOTH sides carry no signal, not EITHER.
        """
        self.assertFalse(cgo.is_vacuous_pair([0.0, 0.0], [1.0, 2.0]))
        self.assertFalse(cgo.is_vacuous_pair([1.0, 2.0], [0.0, 0.0]))

    def test_both_sides_nonzero_is_not_vacuous_even_if_orthogonal(self):
        """Orthogonal-but-nonzero (a real directional disagreement, cosine
        0.0 for a DIFFERENT reason than "no signal") must not be swept into
        the vacuous bucket either.
        """
        self.assertFalse(cgo.is_vacuous_pair([1.0, 0.0], [0.0, 1.0]))

    def test_compare_tensor_reports_vacuous_field(self):
        vacuous_stats = cgo.compare_tensor("lora_a", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertTrue(vacuous_stats["vacuous"])
        self.assertEqual(vacuous_stats["cosine_similarity"], 0.0)

        real_stats = cgo.compare_tensor("lora_b", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertFalse(real_stats["vacuous"])

    def test_compare_reports_counts_and_names_vacuous_tensors_without_excluding_them(self):
        """Models the confirmed-live shape: one `lora_a`-style tensor that
        is exactly zero on both sides (vacuous) alongside one `lora_b`-
        style tensor that carries real, IDENTICAL signal. The vacuous
        tensor must be COUNTED and NAMED, and `overall_cosine_similarity`
        must reflect ONLY the real signal (mathematically unaffected by
        also concatenating a both-zero segment -- see `compare_reports`'s
        own doc for why this is not an exclusion, just an invariant of the
        arithmetic) -- 1.0 here, not diluted toward 0 by the zero segment.
        """
        report_a = make_report(
            0.3,
            {
                "layer.0.Wqkv.lora_a": [0.0, 0.0, 0.0, 0.0],
                "layer.0.Wqkv.lora_b": [1.0, 2.0, 3.0, 4.0],
            },
        )
        report_b = make_report(
            0.3,
            {
                "layer.0.Wqkv.lora_a": [0.0, 0.0, 0.0, 0.0],
                "layer.0.Wqkv.lora_b": [1.0, 2.0, 3.0, 4.0],
            },
        )
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(result["matched_tensor_count"], 2)
        self.assertEqual(result["vacuous_tensor_count"], 1)
        self.assertEqual(result["vacuous_tensor_names"], ["layer.0.Wqkv.lora_a"])
        self.assertEqual(result["overall_cosine_similarity"], 1.0)
        self.assertTrue(result["passed"])

    def test_main_prints_the_vacuous_count(self):
        """Drives `main()` (the real entry point): the vacuous count must
        be VISIBLE in the printed output, not only in the JSON `--out`
        file, so a human reading CI's log sees it. Mirrors the confirmed-
        live shape (a mix of vacuous `lora_a` and real, agreeing `lora_b`
        tensors, not an all-vacuous degenerate case) so this also pins
        that a REAL signal elsewhere still earns a `PASS` alongside a
        nonzero vacuous count -- vacuous is a classification, not an
        automatic fail.
        """
        import contextlib
        import io
        import json as _json
        import os as _os
        import tempfile as _tempfile

        grads = {
            "layer.0.Wqkv.lora_a": [0.0, 0.0],
            "layer.0.Wqkv.lora_b": [1.0, 2.0, 3.0],
        }
        report_a = make_report(0.3, grads)
        report_b = make_report(0.3, grads)
        with _tempfile.TemporaryDirectory() as d:
            pa, pb = _os.path.join(d, "a.json"), _os.path.join(d, "b.json")
            with open(pa, "w") as fh:
                _json.dump(report_a, fh)
            with open(pb, "w") as fh:
                _json.dump(report_b, fh)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cgo.main([pa, pb, "--cosine-floor", "0.9"])
        self.assertIn("vacuous_tensor_count: 1", out.getvalue())
        self.assertEqual(code, cgo.EXIT_PASS)


class SameProducerGuardTests(unittest.TestCase):
    """ITEM 6 of this round's audit: `compare a.json a.json` (or, more
    generally, comparing a producer against itself) used to PASS —
    `compare_reports` never checked that the two dumps came from two
    INDEPENDENT producers at all. `_same_producer_violation` closes this;
    every test here constructs two dumps sharing the SAME `tool` value
    (never relying on `make_report`'s new unique-by-default tool, which is
    exactly the mechanism that keeps every OTHER test in this file from
    tripping this guard by accident).
    """

    def test_same_tool_value_is_refused_by_default(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="same_tool")
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="same_tool")
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertFalse(result["passed"])
        self.assertTrue(any("same tool" in v for v in result["premise_violations"]))

    def test_literal_same_file_compared_to_itself_is_refused_at_main(self):
        """THE LITERAL REPRODUCTION: `compare_grad_oracle.py a.json a.json`
        -- driven at `main()`, the real entry point, with the SAME path
        passed twice.
        """
        report = make_report(0.31, {"t": [0.1, -0.2, 0.3]})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.json")
            with open(path, "w") as fh:
                json.dump(report, fh)
            import contextlib
            import io

            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cgo.main([path, path, "--cosine-floor", "0.9"])
        self.assertEqual(code, cgo.EXIT_FAIL, f"stdout={out.getvalue()!r} stderr={err.getvalue()!r}")
        self.assertNotIn("PASS", out.getvalue().splitlines())
        self.assertIn("PREMISE VIOLATION", err.getvalue())

    def test_allow_same_producer_override_lets_a_genuine_self_check_pass(self):
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="same_tool")
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="same_tool")
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9, allow_same_producer=True)
        self.assertTrue(result["passed"], f"premise_violations={result['premise_violations']!r}")

    def test_allow_same_producer_flag_wires_through_main(self):
        report = make_report(0.31, {"t": [0.1, -0.2, 0.3]})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.json")
            with open(path, "w") as fh:
                json.dump(report, fh)
            import contextlib
            import io

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cgo.main([path, path, "--cosine-floor", "0.9", "--allow-same-producer"])
        self.assertEqual(code, cgo.EXIT_PASS, f"stdout={out.getvalue()!r}")
        self.assertIn("PASS", out.getvalue().splitlines())

    def test_genuinely_different_tools_are_not_flagged(self):
        """Positive control: real producers set DIFFERENT literal `tool`
        strings by construction (`jammi_grad_oracle` vs `torch_grad_oracle`)
        -- this guard must never fire on a genuine cross-framework pair.
        """
        report_a = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="jammi_grad_oracle")
        report_b = make_report(0.3, {"t": [1.0, 2.0, 3.0]}, tool="torch_grad_oracle")
        result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
        self.assertEqual(
            [v for v in result["premise_violations"] if "same tool" in v], [],
        )


@unittest.skipUnless(
    cgo.HAVE_NUMPY,
    "numpy is not importable in this environment -- this class compares the numpy arm "
    "against the pure-Python fallback arm on the SAME divergent inputs; with no numpy, there "
    "is only one arm to compare, so arm PARITY cannot be exercised here. NOTE (item 4 of this "
    "round's audit): this repo's CI Guard job (.github/workflows/ci.yml's `guard` matrix, the "
    "lane that runs this file) installs NO python packages at all before invoking `python3` -- "
    "no actions/setup-python with a requirements file, no `pip install numpy` step anywhere in "
    "that job -- so on a stock ubuntu-latest runner's system python3 this class is expected to "
    "SKIP there too. Treat 'the numpy arm's own correctness' as verified ONLY in an environment "
    "that separately provisions numpy (e.g. $TORCH_VENV, which pulls it in as torch's own "
    "transitive dependency) -- do not claim CI Guard coverage this class does not actually run.",
)
class ArmParityTests(unittest.TestCase):
    """ITEM 4 of this round's audit: the numpy arm and the pure-Python
    fallback arm are never asserted equal on the SAME divergent input --
    the "run twice" pattern other tests use (`ComparatorMathTestsMixin`'s
    two concrete subclasses) each runs the FULL suite under exactly ONE
    arm; neither ever compares the two arms' OUTPUT on the same input
    within a single test. This class does: every test here calls the SAME
    helper under `HAVE_NUMPY = True` and then `HAVE_NUMPY = False`
    (explicitly toggled, restored in `tearDown` -- never silently reusing
    one arm's result for both, which would make a "ran both arms" test
    into a "ran one arm twice" test without anyone noticing) and asserts
    the two results agree.
    """

    def setUp(self):
        self._orig = cgo.HAVE_NUMPY
        self.assertTrue(self._orig, "guarded by skipUnless(cgo.HAVE_NUMPY, ...) above")

    def tearDown(self):
        cgo.HAVE_NUMPY = self._orig

    def _numpy_arm(self, fn, *args):
        cgo.HAVE_NUMPY = True
        return fn(*args)

    def _pure_arm(self, fn, *args):
        cgo.HAVE_NUMPY = False
        return fn(*args)

    def test_cosine_similarity_agrees_on_a_nan_poisoned_pair(self):
        a, b = [float("nan"), 1.0, 2.0], [1.0, 1.0, 1.0]
        numpy_result = self._numpy_arm(cgo.cosine_similarity, a, b)
        pure_result = self._pure_arm(cgo.cosine_similarity, a, b)
        self.assertTrue(math.isnan(numpy_result), f"numpy arm: {numpy_result}")
        self.assertTrue(math.isnan(pure_result), f"pure arm: {pure_result}")

    def test_has_nonfinite_agrees_on_a_nan_poisoned_pair(self):
        a = [float("nan"), 1.0]
        numpy_result = self._numpy_arm(cgo._has_nonfinite, a)
        pure_result = self._pure_arm(cgo._has_nonfinite, a)
        self.assertEqual(numpy_result, pure_result)
        self.assertTrue(numpy_result)

    def test_has_nonfinite_agrees_on_a_finite_pair(self):
        a = [1.0, 2.0, 3.0]
        self.assertEqual(self._numpy_arm(cgo._has_nonfinite, a), self._pure_arm(cgo._has_nonfinite, a))
        self.assertFalse(self._numpy_arm(cgo._has_nonfinite, a))

    def test_dot_raises_valueerror_on_length_mismatch_in_both_arms(self):
        """THE REPRODUCTION this closes: `zip(a, b)` in the pure-Python arm
        used to SILENTLY TRUNCATE to the shorter length on a length
        mismatch instead of raising -- numpy's own arithmetic raises for a
        genuine shape mismatch. `_require_same_length` now makes both arms
        raise `ValueError` identically, checked BEFORE either arm's own
        branch.
        """
        a, b = [1.0, 2.0], [1.0]
        with self.assertRaises(ValueError):
            self._numpy_arm(cgo._dot, a, b)
        with self.assertRaises(ValueError):
            self._pure_arm(cgo._dot, a, b)

    def test_max_abs_delta_raises_valueerror_on_length_mismatch_in_both_arms(self):
        a, b = [1.0, 2.0, 3.0], [1.0, 2.0]
        with self.assertRaises(ValueError):
            self._numpy_arm(cgo._max_abs_delta, a, b)
        with self.assertRaises(ValueError):
            self._pure_arm(cgo._max_abs_delta, a, b)

    def test_max_abs_delta_length_one_vs_n_does_not_silently_broadcast(self):
        """The BROADCASTING variant of the same reproduction: numpy's own
        elementwise `a - b` would silently SUCCEED (no raise at all) for a
        length-1-vs-length-N pair via ordinary numpy broadcasting rules --
        arguably worse than truncation, since it produces a plausible-
        looking number instead of an obvious crash. `_require_same_length`
        closes this for the numpy arm too, not just the pure-Python one.
        """
        a, b = [1.0], [1.0, 2.0, 3.0]
        with self.assertRaises(ValueError):
            self._numpy_arm(cgo._max_abs_delta, a, b)
        with self.assertRaises(ValueError):
            self._pure_arm(cgo._max_abs_delta, a, b)

    def test_weight_max_violation_agrees_on_a_nan_weight_in_both_arms(self):
        wa = [1.0, float("nan")]
        wb = [1.0, 1.0]
        numpy_bad, numpy_max = self._numpy_arm(cgo._weight_max_violation, wa, wb)
        pure_bad, pure_max = self._pure_arm(cgo._weight_max_violation, wa, wb)
        self.assertGreaterEqual(numpy_bad, 1)
        self.assertGreaterEqual(pure_bad, 1)
        self.assertTrue(math.isnan(numpy_max))
        self.assertTrue(math.isnan(pure_max))

    def test_weight_max_violation_agrees_on_a_finite_agreeing_pair(self):
        wa = [1.0, 2.0, 3.0]
        wb = [1.0, 2.0, 3.0]
        numpy_result = self._numpy_arm(cgo._weight_max_violation, wa, wb)
        pure_result = self._pure_arm(cgo._weight_max_violation, wa, wb)
        self.assertEqual(numpy_result[0], 0)
        self.assertEqual(pure_result[0], 0)
        self.assertAlmostEqual(numpy_result[1], pure_result[1], places=9)


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
