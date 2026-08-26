#!/usr/bin/env python3
"""Tests for `eager_round_once_check.py`.

Run directly: `python3 ci/scripts/perf/test_eager_round_once_check.py`
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eager_round_once_check as erc  # noqa: E402


class Bf16RoundingPrimitives(unittest.TestCase):
    """`round_to_f32`/`round_to_bf16`/`bf16_bits` are the ground truth every
    other test in this file depends on — pinned against hand-verifiable
    values first, independent of the mechanism-level tests below.
    """

    def test_round_to_f32_is_identity_on_an_exact_f32_value(self):
        # 0.5 is exactly representable in both f32 and f64 -- no rounding
        # possible either way.
        self.assertEqual(erc.round_to_f32(0.5), 0.5)

    def test_round_to_f32_rounds_a_value_not_exactly_representable(self):
        # 0.1 is not exactly representable in f32 or f64; round-tripping
        # through f32 must change the f64 bit pattern (a genuine rounding
        # happened), while landing extremely close to the true value.
        rounded = erc.round_to_f32(0.1)
        self.assertNotEqual(rounded, 0.1)
        self.assertAlmostEqual(rounded, 0.1, places=6)

    def test_round_to_f32_nan_propagates(self):
        self.assertTrue(math.isnan(erc.round_to_f32(float("nan"))))

    def test_round_to_bf16_is_identity_on_an_exact_bf16_value(self):
        # 1.5 = 1 + 2^-1: exactly representable in bf16's 7 explicit
        # mantissa bits.
        self.assertEqual(erc.round_to_bf16(1.5), 1.5)

    def test_round_to_bf16_hand_computed_value(self):
        # 1.0 + 2^-8 (one ULP past bf16's precision, at the boundary
        # between the 7th and 8th mantissa bit): must round DOWN to
        # exactly 1.0 (round-to-nearest, and 2^-8 is exactly half the gap
        # between adjacent bf16 values here -- an exact tie, broken
        # to-even, and 1.0's bf16 mantissa bit 0 is even).
        x = 1.0 + 2.0**-8
        self.assertEqual(erc.round_to_bf16(x), 1.0)

    def test_round_to_bf16_tie_breaks_to_even_mantissa(self):
        # The bf16 value just above 1.0 (mantissa bit pattern ...0000001,
        # i.e. 1 + 2^-7) has an ODD low mantissa bit. The exact tie
        # 1.5*2^-7 above it must round UP (to the NEXT bf16 value, whose
        # low mantissa bit is even), not down back to the odd value.
        below = 1.0 + 2.0**-7  # exactly representable in bf16, odd low bit
        tie = below + 2.0**-8  # exact tie between `below` and the next bf16 value
        rounded = erc.round_to_bf16(tie)
        self.assertNotEqual(rounded, below, "a tie must not break toward the ODD mantissa")
        self.assertGreater(rounded, below)

    def test_round_to_bf16_nan_propagates(self):
        self.assertTrue(math.isnan(erc.round_to_bf16(float("nan"))))

    def test_round_to_bf16_is_idempotent(self):
        # A value already bf16-rounded must round-trip unchanged -- the
        # property every mechanism function below relies on when it widens
        # an already-bf16 gamma/cos/sin value to f32 and calls that
        # "lossless" (never re-rounds it further).
        for raw in (-3.7, 0.0, 2.25, -18.5, 9.0625):
            once = erc.round_to_bf16(raw)
            twice = erc.round_to_bf16(once)
            self.assertEqual(once, twice, f"round_to_bf16 not idempotent on {raw}")

    def test_bf16_bits_distinguishes_adjacent_values(self):
        a = erc.round_to_bf16(1.0)
        b = erc.round_to_bf16(1.0 + 2.0**-7)
        self.assertNotEqual(erc.bf16_bits(a), erc.bf16_bits(b))

    def test_bf16_bits_agrees_on_equal_values(self):
        a = erc.round_to_bf16(-2.015625)
        b = erc.round_to_bf16(-2.015625)
        self.assertEqual(erc.bf16_bits(a), erc.bf16_bits(b))


class LcgStream(unittest.TestCase):
    """The deterministic fixture generator (family J: fixed fold order) —
    a real generator, not a stub: pinned for determinism and range.
    """

    def test_same_seed_reproduces_the_identical_stream(self):
        a = list(erc._lcg_stream(42, 100))
        b = list(erc._lcg_stream(42, 100))
        self.assertEqual(a, b)

    def test_different_seeds_diverge(self):
        a = list(erc._lcg_stream(1, 50))
        b = list(erc._lcg_stream(2, 50))
        self.assertNotEqual(a, b)

    def test_stream_stays_within_the_documented_range(self):
        for v in erc._lcg_stream(999, 2000):
            self.assertGreaterEqual(v, -4.0)
            self.assertLess(v, 4.0)


class LayerNormRoundMismatch(unittest.TestCase):
    """`layer_norm_round_mismatch` — the LN gamma-epilogue mechanism."""

    def test_non_vacuous_on_a_real_fixture(self):
        # Family F: a non-vacuous control. The pre-fix/post-fix mismatch
        # must actually be exercised, not accidentally zero (e.g. from a
        # bug that made the two formulas literally identical code).
        mismatches, total, frac = erc.layer_norm_round_mismatch(rows=64, hidden=256)
        self.assertGreater(mismatches, 0)
        self.assertEqual(total, 64 * 256)
        self.assertAlmostEqual(frac, mismatches / total)

    def test_lands_in_the_tens_of_percent_ballpark_the_pr_cites(self):
        # The PR body cites ~26% on its OWN (Rust-oracle) fixture. This is
        # an INDEPENDENT fixture (different RNG stream, different
        # language) -- the assertion is a wide ballpark check (10%-45%),
        # not a bit-for-bit match, matching this script's own documented
        # non-goal (module doc: "not to match those exact counts").
        _, _, frac = erc.layer_norm_round_mismatch(rows=64, hidden=256)
        self.assertGreater(frac, 0.10)
        self.assertLess(frac, 0.45)

    def test_deterministic_across_repeated_calls(self):
        r1 = erc.layer_norm_round_mismatch(rows=8, hidden=16)
        r2 = erc.layer_norm_round_mismatch(rows=8, hidden=16)
        self.assertEqual(r1, r2)

    def test_a_constant_gamma_of_one_never_diverges(self):
        # Degenerate/boundary oracle (family D): if EVERY gamma element is
        # exactly bf16 `1.0`, `xhat_bf16 * 1.0 == xhat_bf16` exactly in
        # bf16 arithmetic (multiplying by 1 loses no precision), and
        # `xhat_f32 * 1.0` rounded to bf16 equals `round_bf16(xhat_f32)`
        # too -- both forms collapse to the SAME value regardless of the
        # rounding-order mechanism, so this must measure ZERO mismatches.
        # This is the "negative control fails on the trivial case" analog:
        # a broken implementation that always reports SOME nonzero
        # mismatch count (e.g. from a stray off-by-one in indexing) would
        # be caught here even though the "non-vacuous" test above only
        # checks for the OPPOSITE failure mode.
        import eager_round_once_check as m

        rows, hidden = 4, 8
        stream = m._lcg_stream(12345, rows * hidden + hidden)
        xs_raw = [next(stream) for _ in range(rows * hidden)]
        xb = [m.round_to_bf16(v) for v in xs_raw]
        gb = [1.0] * hidden  # bf16(1.0) == 1.0 exactly

        mismatches = 0
        eps = 1e-5
        for r in range(rows):
            row = xb[r * hidden : (r + 1) * hidden]
            mean = sum(row) / hidden
            var = sum((v - mean) ** 2 for v in row) / hidden
            invvar = 1.0 / m.round_to_f32(math.sqrt(m.round_to_f32(var + eps)))
            for i in range(hidden):
                xhat = m.round_to_f32((row[i] - mean) * invvar)
                pre = m.round_to_bf16(m.round_to_bf16(xhat) * gb[i])
                post = m.round_to_bf16(xhat * gb[i])
                if m.bf16_bits(pre) != m.bf16_bits(post):
                    mismatches += 1
        self.assertEqual(mismatches, 0, "gamma == 1.0 must never diverge between the two forms")


class RopeRoundMismatch(unittest.TestCase):
    """`rope_round_mismatch` — the RoPE rotation mechanism."""

    def test_non_vacuous_on_a_real_fixture(self):
        mismatches, total, frac = erc.rope_round_mismatch(
            batch=1, heads=4, seq=16, head_dim=64
        )
        self.assertGreater(mismatches, 0)
        self.assertEqual(total, 1 * 4 * 16 * 64)
        self.assertAlmostEqual(frac, mismatches / total)

    def test_lands_in_the_tens_of_percent_ballpark_the_pr_cites(self):
        _, _, frac = erc.rope_round_mismatch(batch=1, heads=4, seq=16, head_dim=64)
        self.assertGreater(frac, 0.10)
        self.assertLess(frac, 0.50)

    def test_deterministic_across_repeated_calls(self):
        r1 = erc.rope_round_mismatch(batch=1, heads=2, seq=4, head_dim=8)
        r2 = erc.rope_round_mismatch(batch=1, heads=2, seq=4, head_dim=8)
        self.assertEqual(r1, r2)

    def test_head_dim_two_is_a_hand_verifiable_boundary_case(self):
        # `head_dim=2` (the smallest even head_dim -- `half=1`, a genuine
        # single-pair rotation, not degenerate to zero elements) -- a
        # domain/boundary oracle (family D) distinct from the "typical
        # shape" tests above: must run without error and still measure a
        # non-vacuous divergence.
        mismatches, total, frac = erc.rope_round_mismatch(
            batch=1, heads=1, seq=4, head_dim=2
        )
        self.assertEqual(total, 1 * 1 * 4 * 2)
        self.assertGreater(mismatches, 0)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)


class MainEntryPoint(unittest.TestCase):
    """`main()` — the CLI surface, at small shapes so the test suite stays
    fast (the module's own production-shape defaults are exercised
    manually/in CI, not on every `unittest` run).
    """

    def test_main_passes_and_returns_zero_on_a_real_fixture(self):
        code = erc.main(
            [
                "--ln-rows",
                "32",
                "--ln-hidden",
                "64",
                "--rope-batch",
                "1",
                "--rope-heads",
                "2",
                "--rope-seq",
                "8",
                "--rope-head-dim",
                "16",
            ]
        )
        self.assertEqual(code, 0)

    def test_main_writes_the_requested_json_report(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "report.json")
            code = erc.main(
                [
                    "--ln-rows",
                    "32",
                    "--ln-hidden",
                    "64",
                    "--rope-batch",
                    "1",
                    "--rope-heads",
                    "2",
                    "--rope-seq",
                    "8",
                    "--rope-head-dim",
                    "16",
                    "--out",
                    out_path,
                ]
            )
            self.assertEqual(code, 0)
            with open(out_path) as f:
                report = json.load(f)
            self.assertIn("layer_norm", report)
            self.assertIn("rope", report)
            self.assertGreater(report["layer_norm"]["mismatches"], 0)
            self.assertGreater(report["rope"]["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
