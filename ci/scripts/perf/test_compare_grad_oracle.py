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

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_grad_oracle as cgo  # noqa: E402


def make_report(loss, gradients, **overrides):
    """`gradients`: {name: [f32, ...]}.

    Fills in a MATCHING, premise-satisfying run-identity/weight-provenance
    envelope by default (`lora_weights_in` set, standard seed/batch/etc.,
    `batch_token_id_sums` present) — F3's fix made `compare_reports` check
    all of these, so a test that only cares about the gradient-DIRECTION
    math (the vast majority below) must not ALSO have to hand-build a full
    premise or it would spuriously fail on a premise violation it never
    meant to test. Pass e.g. `lora_weights_in=None` or `seed=7` via
    `overrides` to specifically construct a premise violation (see
    `PremiseAndWeightChecks` below).
    """
    report = {
        "tool": "test-fixture",
        "loss": loss,
        "seed": 1,
        "batch": 2,
        "seq": 4,
        "lora_rank": 2,
        "target_modules": ["Wqkv"],
        "batched_forward": True,
        "backbone_dtype": "f32",
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
            ("target_modules", ["Wo"]),
            ("batched_forward", False),
            ("backbone_dtype", "bf16"),
        ):
            report_a = make_report(0.3, {"t": [1.0, 2.0]})
            report_b = make_report(0.3, {"t": [1.0, 2.0]}, **{field: other_value})
            result = cgo.compare_reports(report_a, report_b, cosine_floor=0.9)
            self.assertFalse(result["passed"], f"a {field!r} mismatch must refuse")
            self.assertTrue(
                any(field in v for v in result["premise_violations"]),
                f"expected a premise_violations entry naming {field!r}, got {result['premise_violations']!r}",
            )

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
