#!/usr/bin/env python3
"""`gen_fixed_width_corpus.py`'s own suite (P4, CONTRACT
`scratchpad/contract-356-profile.md` v3): determinism (same
`(rows, min_wordpieces, seed)` -> byte-identical output; different seed ->
different output), the emitted JSONL schema (exactly the six
`anchor_id`/`anchor_text`/`positive_id`/`positive_text`/`negative_id`/
`negative_text` keys `crates/jammi-bench/src/main.rs::TripletRow` expects,
pinned by a literal field-name check, never a re-derived guess), and the
width-margin guarantee (every emitted text has strictly more
whitespace-separated words than `--min-wordpieces`, which is this
generator's own mechanism for exceeding the wordpiece cap under a
whitespace-pre-splitting tokenizer -- see the module doc's "Construction
and its guarantee").

Stdlib-only (`unittest`), no network, no `tokenizers` package required
(the `--verify-tokenizer` mechanical check is exercised separately, gated
behind the package's availability).

Run: `python3 ci/scripts/perf/test_gen_fixed_width_corpus.py`
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_fixed_width_corpus as gfw  # noqa: E402

_EXPECTED_KEYS = {
    "anchor_id",
    "anchor_text",
    "positive_id",
    "positive_text",
    "negative_id",
    "negative_text",
}


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_byte_identical(self):
        rows_a = gfw.generate_rows(rows=20, min_wordpieces=8, seed=7)
        rows_b = gfw.generate_rows(rows=20, min_wordpieces=8, seed=7)
        self.assertEqual(rows_a, rows_b)

    def test_different_seed_differs(self):
        rows_a = gfw.generate_rows(rows=20, min_wordpieces=8, seed=7)
        rows_b = gfw.generate_rows(rows=20, min_wordpieces=8, seed=8)
        self.assertNotEqual(rows_a, rows_b)

    def test_smaller_rows_is_a_prefix_of_larger_rows(self):
        small = gfw.generate_rows(rows=5, min_wordpieces=6, seed=3)
        large = gfw.generate_rows(rows=17, min_wordpieces=6, seed=3)
        self.assertEqual(small, large[:5])

    def test_write_jsonl_round_trip_is_deterministic(self):
        rows = gfw.generate_rows(rows=6, min_wordpieces=5, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            out1 = Path(tmp) / "a.jsonl"
            out2 = Path(tmp) / "b.jsonl"
            gfw.write_jsonl(rows, out1)
            gfw.write_jsonl(rows, out2)
            self.assertEqual(out1.read_bytes(), out2.read_bytes())


class SchemaTests(unittest.TestCase):
    def test_every_row_has_exactly_the_six_expected_keys(self):
        rows = gfw.generate_rows(rows=10, min_wordpieces=4, seed=1)
        for row in rows:
            self.assertEqual(set(row.keys()), _EXPECTED_KEYS)
            for key in _EXPECTED_KEYS:
                self.assertIsInstance(row[key], str)
                self.assertTrue(row[key])

    def test_written_jsonl_is_one_valid_json_object_per_line(self):
        rows = gfw.generate_rows(rows=4, min_wordpieces=3, seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus.jsonl"
            gfw.write_jsonl(rows, out)
            lines = out.read_text().splitlines()
            self.assertEqual(len(lines), 4)
            for line, expected in zip(lines, rows, strict=True):
                parsed = json.loads(line)
                self.assertEqual(parsed, expected)

    def test_ids_are_unique_across_rows_and_roles(self):
        rows = gfw.generate_rows(rows=15, min_wordpieces=4, seed=9)
        ids = []
        for row in rows:
            ids.extend([row["anchor_id"], row["positive_id"], row["negative_id"]])
        self.assertEqual(
            len(ids), len(set(ids)), "expected every id across every row/role to be unique"
        )


class WidthMarginTests(unittest.TestCase):
    def test_every_text_has_more_than_min_wordpieces_whitespace_tokens(self):
        for min_wp in (0, 1, 4, 64, 512):
            rows = gfw.generate_rows(rows=3, min_wordpieces=min_wp, seed=5)
            for row in rows:
                for key in ("anchor_text", "positive_text", "negative_text"):
                    word_count = len(row[key].split(" "))
                    self.assertGreater(
                        word_count, min_wp,
                        f"min_wordpieces={min_wp}, text={row[key]!r} has only {word_count} words",
                    )

    def test_words_are_drawn_only_from_the_fixed_vocab(self):
        rows = gfw.generate_rows(rows=5, min_wordpieces=6, seed=13)
        vocab = set(gfw._VOCAB)
        for row in rows:
            for key in ("anchor_text", "positive_text", "negative_text"):
                for word in row[key].split(" "):
                    self.assertIn(word, vocab)


class ValidationTests(unittest.TestCase):
    def test_nonpositive_rows_refused(self):
        with self.assertRaises(ValueError):
            gfw.generate_rows(rows=0, min_wordpieces=4, seed=1)
        with self.assertRaises(ValueError):
            gfw.generate_rows(rows=-1, min_wordpieces=4, seed=1)

    def test_negative_min_wordpieces_refused(self):
        with self.assertRaises(ValueError):
            gfw.generate_rows(rows=3, min_wordpieces=-1, seed=1)


class CliTests(unittest.TestCase):
    def test_main_writes_expected_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus.jsonl"
            rc = gfw.main(
                ["--rows", "9", "--min-wordpieces", "5", "--seed", "3", "--out", str(out)]
            )
            self.assertEqual(rc, 0)
            lines = out.read_text().splitlines()
            self.assertEqual(len(lines), 9)


@unittest.skipUnless(
    importlib.util.find_spec("tokenizers") is not None, "tokenizers package not installed"
)
class VerifyTokenizerTests(unittest.TestCase):
    """Only runs when the optional `tokenizers` package is importable --
    exercises the mechanical single-wordpiece verification path against
    whatever tokenizer.json fixture is available. Skipped (never silently
    passed as though it ran) when the package is absent -- the generator's
    core determinism/schema guarantees above do not depend on this."""

    def test_verify_against_a_real_tokenizer_fixture(self):
        repo_root = Path(__file__).resolve().parents[3]
        tokenizer_json = repo_root / "cookbook" / "fixtures" / "tiny_bert" / "tokenizer.json"
        if not tokenizer_json.exists():
            self.skipTest(f"{tokenizer_json} not present in this checkout")
        # A tiny fixture tokenizer's vocab may not cover every _VOCAB word --
        # this just exercises that the function runs and returns a list
        # (empty or not), never that it raises.
        result = gfw.verify_vocab_is_single_wordpiece(tokenizer_json)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
