#!/usr/bin/env python3
"""`fixture_width_report.py`'s own suite (P4, CONTRACT
`scratchpad/contract-356-profile.md` v3 Artifacts / "Width pinning"):
drives the real `derive_claim` pigeonhole-verdict logic against a synthetic
`reaches_cap` length table -- never a re-implementation of the math, and
never requiring the `tokenizers` package (the verdict logic is a PURE
function of the per-row booleans + batch size, exactly so it can be
tested without a real tokenizer -- see that module's own doc).

The `tokenizers`-dependent half (`per_row_lengths`/`build_report`'s own
tokenizer calls, and the whole-module `--tokenizer`-not-importable loud
refusal) is exercised only when `tokenizers` is actually installed --
skipped, never silently treated as passing, otherwise.

Run: `python3 ci/scripts/perf/test_fixture_width_report.py`
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixture_width_report as fwr  # noqa: E402

PERF_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(PERF_DIR, "fixture_width_report.py")


class DeriveClaimPigeonholeTests(unittest.TestCase):
    def test_all_rows_reach_cap_k_is_zero_any_batch_size_is_uniform(self):
        claim = fwr.derive_claim([True, True, True, True], batch_size=1)
        self.assertEqual(claim["k_short_rows"], 0)
        self.assertEqual(claim["r_threshold"], 1)
        self.assertEqual(claim["verdict"], "width uniform at W")

    def test_no_rows_reach_cap_k_equals_n_never_provable_below_n_plus_1(self):
        claim = fwr.derive_claim([False, False, False], batch_size=3)
        self.assertEqual(claim["k_short_rows"], 3)
        self.assertEqual(claim["r_threshold"], 4)
        self.assertEqual(claim["verdict"], "not provable")

    def test_mixed_k_short_rows_threshold_is_k_plus_one(self):
        # 12 rows, 3 short (reaches_cap=False) -- r_threshold = 4.
        # batch_size=6 is chosen so 12 % 6 == 0 (no remainder batch) --
        # this test is about the base r_threshold=k+1 arithmetic, not the
        # remainder-batch qualification (see RemainderBatchTests below).
        reaches_cap = [False, True, True, False, True, True, True, False, True, True, True, True]
        claim = fwr.derive_claim(reaches_cap, batch_size=6)
        self.assertEqual(claim["n_rows"], 12)
        self.assertEqual(claim["k_short_rows"], 3)
        self.assertEqual(claim["r_threshold"], 4)
        self.assertEqual(claim["remainder_batch_size"], 0)
        self.assertEqual(claim["verdict"], "width uniform at W")

    def test_batch_size_exactly_k_is_not_provable(self):
        reaches_cap = [False, False, True, True, True]
        claim = fwr.derive_claim(reaches_cap, batch_size=2)  # batch_size == k
        self.assertEqual(claim["k_short_rows"], 2)
        self.assertEqual(claim["verdict"], "not provable")

    def test_batch_size_one_more_than_k_is_uniform(self):
        # 6 rows (2 short), batch_size=3: 6 % 3 == 0 -- no remainder batch,
        # so the base "batch_size == k+1" case gives an unconditional
        # verdict (see RemainderBatchTests for the case where it does not).
        reaches_cap = [False, False, True, True, True, True]
        claim = fwr.derive_claim(reaches_cap, batch_size=3)  # batch_size == k+1
        self.assertEqual(claim["remainder_batch_size"], 0)
        self.assertEqual(claim["verdict"], "width uniform at W")

    def test_nonpositive_batch_size_refused(self):
        with self.assertRaises(ValueError):
            fwr.derive_claim([True, False], batch_size=0)
        with self.assertRaises(ValueError):
            fwr.derive_claim([True, False], batch_size=-1)

    def test_empty_row_set(self):
        claim = fwr.derive_claim([], batch_size=1)
        self.assertEqual(claim["n_rows"], 0)
        self.assertEqual(claim["k_short_rows"], 0)
        self.assertEqual(claim["r_threshold"], 1)
        self.assertEqual(claim["remainder_batch_size"], 0)
        self.assertEqual(claim["verdict"], "width uniform at W")


class RemainderBatchTests(unittest.TestCase):
    """The remainder-batch qualification (module doc "THE REMAINDER
    BATCH"): a corpus whose row count is not an exact multiple of
    `batch_size` has one final batch SMALLER than the rest, which needs
    its own, stricter threshold check -- never silently folded into an
    unconditional "uniform" verdict."""

    def test_remainder_smaller_than_threshold_qualifies_the_verdict(self):
        # 10 rows, batch_size=4: 10 % 4 == 2 (remainder batch of size 2).
        # k=3 -- full-size (4-row) batches clear r_threshold=4, but the
        # 2-row remainder batch does NOT (2 < 4) -- QUALIFIED verdict,
        # never unconditional "uniform".
        reaches_cap = [False, True, True, False, True, True, True, False, True, True]
        claim = fwr.derive_claim(reaches_cap, batch_size=4)
        self.assertEqual(claim["n_rows"], 10)
        self.assertEqual(claim["k_short_rows"], 3)
        self.assertEqual(claim["r_threshold"], 4)
        self.assertEqual(claim["remainder_batch_size"], 2)
        self.assertNotEqual(claim["verdict"], "width uniform at W")
        self.assertIn("full-size", claim["verdict"])
        self.assertIn("remainder batch", claim["verdict"])
        self.assertIn("not provable", claim["verdict"])

    def test_remainder_at_least_threshold_is_still_unconditionally_uniform(self):
        # 5 rows, batch_size=3: 5 % 3 == 2 (remainder batch of size 2).
        # k=0 -- r_threshold=1, so BOTH the 3-row full batches AND the
        # 2-row remainder batch clear it (2 >= 1) -- unconditional
        # "width uniform at W", not qualified.
        reaches_cap = [True, True, True, True, True]
        claim = fwr.derive_claim(reaches_cap, batch_size=3)
        self.assertEqual(claim["remainder_batch_size"], 2)
        self.assertEqual(claim["verdict"], "width uniform at W")

    def test_full_batches_not_uniform_remainder_cannot_rescue_it(self):
        # batch_size itself fails the threshold (batch_size == k) -- the
        # remainder cannot make this any more provable, whatever its size.
        reaches_cap = [False, False, True, True, True, True, True]  # n=7, k=2
        claim = fwr.derive_claim(reaches_cap, batch_size=2)  # 7 % 2 == 1
        self.assertEqual(claim["remainder_batch_size"], 1)
        self.assertEqual(claim["verdict"], "not provable")

    def test_e1_like_1372_pairs_batch_32_remainder_28_qualifies(self):
        # The contract's own real numbers (module doc): 1372 train pairs,
        # batch 32, 1372 % 32 == 28. k=30 short rows -- full-size (32-row)
        # batches clear r_threshold=31 (32 >= 31), but the 28-row remainder
        # batch does not (28 < 31) -- QUALIFIED, the exact class this
        # fixture-shaped test pins.
        n, batch_size, k = 1372, 32, 30
        self.assertEqual(n % batch_size, 28)
        reaches_cap = [False] * k + [True] * (n - k)
        claim = fwr.derive_claim(reaches_cap, batch_size=batch_size)
        self.assertEqual(claim["remainder_batch_size"], 28)
        self.assertEqual(claim["r_threshold"], 31)
        self.assertNotEqual(claim["verdict"], "width uniform at W")
        self.assertIn("not provable", claim["verdict"])

    def test_e1_like_1372_pairs_batch_32_remainder_28_still_uniform_when_k_is_small(self):
        # Same real shape, but few enough short rows (k=10) that even the
        # 28-row remainder batch clears r_threshold=11 -- unconditional
        # "width uniform at W", no qualification needed.
        n, batch_size, k = 1372, 32, 10
        reaches_cap = [False] * k + [True] * (n - k)
        claim = fwr.derive_claim(reaches_cap, batch_size=batch_size)
        self.assertEqual(claim["remainder_batch_size"], 28)
        self.assertEqual(claim["r_threshold"], 11)
        self.assertEqual(claim["verdict"], "width uniform at W")


class BuildReportShapeTests(unittest.TestCase):
    """Exercises `build_report`/`per_row_lengths` against a FAKE tokenizer
    stub (never the real `tokenizers` package) -- pins the SHAPE of what
    `per_row_lengths` reads from a row dict and writes to its own output,
    independent of whether the real package is installed."""

    class _FakeEncoding:
        def __init__(self, ids):
            self.ids = ids

    class _FakeTokenizer:
        """`raw_len_by_text` maps text -> untruncated wordpiece count;
        `cap_reached_by_text` maps text -> whether truncation at the
        configured cap yields exactly `cap` tokens."""

        def __init__(self, raw_len_by_text, cap_reached_by_text):
            self._raw = raw_len_by_text
            self._reaches = cap_reached_by_text
            self._truncating = False
            self._cap = None

        def enable_truncation(self, max_length):
            self._truncating = True
            self._cap = max_length

        def no_truncation(self):
            self._truncating = False
            self._cap = None

        def encode(self, text, add_special_tokens=True):
            if self._truncating:
                specials = 2 if add_special_tokens else 0
                n = self._cap if self._reaches[text] else self._raw[text] + specials
                return BuildReportShapeTests._FakeEncoding([0] * n)
            n = self._raw[text]
            return BuildReportShapeTests._FakeEncoding([0] * n)

    def test_build_report_shape_and_claim_agree_with_per_row_reaches_cap(self):
        pairs = [
            {"anchor_id": "r0", "anchor_text": "short anchor",
             "positive_text": "short positive", "negative_text": "short negative"},
            {"anchor_id": "r1", "anchor_text": "long anchor reaches cap",
             "positive_text": "short positive 2", "negative_text": "short negative 2"},
        ]
        raw = {
            "short anchor": 2, "short positive": 2, "short negative": 2,
            "long anchor reaches cap": 64, "short positive 2": 3, "short negative 2": 3,
        }
        reaches = {
            "short anchor": False, "short positive": False, "short negative": False,
            "long anchor reaches cap": True, "short positive 2": False, "short negative 2": False,
        }
        tok = self._FakeTokenizer(raw, reaches)
        report = fwr.build_report(pairs, tok, cap=64, batch_size=2)
        self.assertEqual(len(report["rows"]), 2)
        # row 0: neither anchor nor positive reaches cap; row 1: anchor reaches cap.
        self.assertFalse(report["rows"][0]["reaches_cap"])
        self.assertTrue(report["rows"][1]["reaches_cap"])
        self.assertEqual(report["claim"]["k_short_rows"], 1)
        self.assertEqual(report["claim"]["r_threshold"], 2)
        self.assertEqual(report["claim"]["verdict"], "width uniform at W")


class NotImportableRefusalTests(unittest.TestCase):
    def test_missing_tokenizers_package_refuses_loudly_no_report(self):
        """Simulates the "tokenizers not importable" path via a real
        subprocess with `tokenizers` hidden from `sys.path` resolution --
        skipped if the real package IS installed in this environment
        (there is then nothing to hide it behind without also breaking
        other tests' imports)."""
        if importlib.util.find_spec("tokenizers") is not None:
            self.skipTest(
                "tokenizers package is installed in this environment; cannot exercise "
                "the ImportError path without uninstalling it"
            )
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = Path(tmp) / "pairs.jsonl"
            pairs_path.write_text(
                json.dumps({"anchor_id": "a", "anchor_text": "x", "positive_id": "p",
                            "positive_text": "y", "negative_id": "n", "negative_text": "z"}) + "\n"
            )
            tokenizer_path = Path(tmp) / "tokenizer.json"
            tokenizer_path.write_text("{}")
            out_path = Path(tmp) / "report.json"
            result = subprocess.run(
                [sys.executable, SCRIPT, str(pairs_path), "--tokenizer", str(tokenizer_path),
                 "--cap", "64", "--out", str(out_path)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_path.exists())


if __name__ == "__main__":
    unittest.main()
