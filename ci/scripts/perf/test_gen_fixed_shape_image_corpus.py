#!/usr/bin/env python3
"""`gen_fixed_shape_image_corpus.py`'s own suite (issue #421 W2b): the SHAPE
guarantee read back off the emitted PNG bytes, determinism (family J), the
emitted JSONL schema pinned by a literal field-name check against
`crates/jammi-bench/src/main.rs::MediaTripletRow`, the triplet separation
asserted MECHANICALLY (intra-family pixel distance strictly below
inter-family), and the input-validation refusals.

Every shape/content assertion goes through `decode_png_rgb` -- i.e. through
the BYTES actually written -- never through the in-memory array the encoder
was handed, so a bug in the encoder cannot be masked by a test that only
inspects the generator's inputs.

Stdlib-only (`unittest`), no network, no Pillow, no numpy.

Run: `python3 -m pytest ci/scripts/perf/test_gen_fixed_shape_image_corpus.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_fixed_shape_image_corpus as gfi  # noqa: E402

# The exact keys `crates/jammi-bench/src/main.rs::MediaTripletRow`
# deserializes. Pinned as a literal set, never re-derived from the producer
# -- a test that asked the producer what it emits would agree with itself
# after any drift.
_EXPECTED_KEYS = {
    "anchor_id",
    "anchor_path",
    "positive_id",
    "positive_path",
    "negative_id",
    "negative_path",
}


def _mean_abs_diff(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        raise AssertionError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


class ShapeTests(unittest.TestCase):
    def test_every_emitted_png_is_exactly_size_by_size_rgb(self):
        for size in (8, 16, 33):
            files, _rows = gfi.generate_corpus(rows=4, size=size, seed=5)
            self.assertTrue(files)
            for name, data in files.items():
                w, h, pixels = gfi.decode_png_rgb(data)
                self.assertEqual((w, h), (size, size), f"{name} is {w}x{h}, expected {size}x{size}")
                self.assertEqual(len(pixels), size * size * 3, name)

    def test_decoder_refuses_a_non_png(self):
        with self.assertRaises(ValueError):
            gfi.decode_png_rgb(b"not a png at all")

    def test_decoder_refuses_a_corrupted_crc(self):
        files, _rows = gfi.generate_corpus(rows=2, size=8, seed=1)
        data = bytearray(next(iter(files.values())))
        # Flip one payload byte inside IHDR; the stored CRC no longer matches.
        data[20] ^= 0xFF
        with self.assertRaises(ValueError):
            gfi.decode_png_rgb(bytes(data))

    def test_encode_png_refuses_a_wrong_length_pixel_buffer(self):
        with self.assertRaises(ValueError):
            gfi.encode_png(4, 4, b"\x00" * 10)


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_byte_identical_files_and_rows(self):
        a_files, a_rows = gfi.generate_corpus(rows=6, size=12, seed=17)
        b_files, b_rows = gfi.generate_corpus(rows=6, size=12, seed=17)
        self.assertEqual(a_files, b_files)
        self.assertEqual(a_rows, b_rows)

    def test_different_seed_changes_the_pixels(self):
        a_files, _ = gfi.generate_corpus(rows=6, size=12, seed=17)
        b_files, _ = gfi.generate_corpus(rows=6, size=12, seed=18)
        self.assertEqual(set(a_files), set(b_files))
        self.assertNotEqual(a_files, b_files)

    def test_rows_count_does_not_perturb_the_image_bytes(self):
        """The row walk consumes no RNG, so a longer corpus reproduces a
        shorter one's images exactly -- the property that lets a profile leg
        scale `--rows` without re-hashing every file."""
        small_files, small_rows = gfi.generate_corpus(rows=3, size=10, seed=4)
        large_files, large_rows = gfi.generate_corpus(rows=11, size=10, seed=4)
        self.assertEqual(small_files, large_files)
        self.assertEqual(small_rows, large_rows[:3])

    def test_written_tree_is_byte_identical_across_two_writes(self):
        files, rows = gfi.generate_corpus(rows=4, size=10, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            gfi.write_corpus(files, rows, a, "triplets.jsonl")
            gfi.write_corpus(files, rows, b, "triplets.jsonl")
            a_names = sorted(p.name for p in a.iterdir())
            b_names = sorted(p.name for p in b.iterdir())
            self.assertEqual(a_names, b_names)
            for name in a_names:
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)


class TripletStructureTests(unittest.TestCase):
    def test_intra_family_distance_is_strictly_below_inter_family(self):
        """The mechanism behind "positive"/"negative", asserted rather than
        assumed: two instances of one family differ only by jitter, two
        instances of different families differ by their whole template."""
        size = 24
        files, _rows = gfi.generate_corpus(
            rows=4, size=size, seed=3, families=4, instances_per_family=3
        )
        px = {name: gfi.decode_png_rgb(data)[2] for name, data in files.items()}

        intra = []
        inter = []
        for f in range(4):
            for i in range(3):
                for j in range(i + 1, 3):
                    intra.append(
                        _mean_abs_diff(px[gfi._image_name(f, i)], px[gfi._image_name(f, j)])
                    )
            for g in range(f + 1, 4):
                for i in range(3):
                    inter.append(
                        _mean_abs_diff(px[gfi._image_name(f, i)], px[gfi._image_name(g, i)])
                    )
        self.assertLess(
            max(intra),
            min(inter),
            f"every intra-family pair must be closer than every inter-family pair; "
            f"max intra={max(intra):.2f}, min inter={min(inter):.2f}",
        )

    def test_every_row_pairs_one_family_against_another(self):
        families = 4
        instances = 4
        _files, rows = gfi.generate_corpus(
            rows=13, size=8, seed=2, families=families, instances_per_family=instances
        )

        def fam_of(path: str) -> int:
            return int(path.split("_f")[1].split("_i")[0])

        for row in rows:
            a = fam_of(row["anchor_path"])
            p = fam_of(row["positive_path"])
            n = fam_of(row["negative_path"])
            self.assertEqual(a, p, f"anchor/positive must share a family: {row}")
            self.assertNotEqual(a, n, f"negative must come from another family: {row}")
            self.assertNotEqual(
                row["anchor_path"], row["positive_path"], f"positive must be a DISTINCT file: {row}"
            )

    def test_two_instances_of_one_family_are_never_byte_identical(self):
        files, _rows = gfi.generate_corpus(rows=2, size=16, seed=6, instances_per_family=2)
        self.assertNotEqual(files[gfi._image_name(0, 0)], files[gfi._image_name(0, 1)])


class SchemaTests(unittest.TestCase):
    def test_every_row_has_exactly_the_six_expected_keys(self):
        _files, rows = gfi.generate_corpus(rows=7, size=8, seed=1)
        for row in rows:
            self.assertEqual(set(row.keys()), _EXPECTED_KEYS)
            for key in _EXPECTED_KEYS:
                self.assertIsInstance(row[key], str)
                self.assertTrue(row[key])

    def test_every_referenced_path_exists_and_is_relative_to_the_jsonl(self):
        files, rows = gfi.generate_corpus(rows=9, size=8, seed=12)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus"
            jsonl = gfi.write_corpus(files, rows, out, "triplets.jsonl")
            for line in jsonl.read_text().splitlines():
                row = json.loads(line)
                for key in ("anchor_path", "positive_path", "negative_path"):
                    rel = Path(row[key])
                    self.assertFalse(rel.is_absolute(), f"{key} must be relative: {row[key]!r}")
                    self.assertTrue((jsonl.parent / rel).is_file(), f"missing {row[key]}")

    def test_ids_are_unique_across_rows_and_roles(self):
        _files, rows = gfi.generate_corpus(rows=15, size=8, seed=9)
        ids = []
        for row in rows:
            ids.extend([row["anchor_id"], row["positive_id"], row["negative_id"]])
        self.assertEqual(len(ids), len(set(ids)))


class ValidationTests(unittest.TestCase):
    def test_nonpositive_rows_refused(self):
        with self.assertRaises(ValueError):
            gfi.generate_corpus(rows=0, size=8, seed=1)

    def test_nonpositive_size_refused(self):
        with self.assertRaises(ValueError):
            gfi.generate_corpus(rows=2, size=0, seed=1)

    def test_one_family_refused(self):
        with self.assertRaises(ValueError):
            gfi.generate_corpus(rows=2, size=8, seed=1, families=1)

    def test_one_instance_per_family_refused(self):
        with self.assertRaises(ValueError):
            gfi.generate_corpus(rows=2, size=8, seed=1, instances_per_family=1)

    def test_zero_jitter_refused(self):
        """A negative control that is non-vacuous: at `jitter == 0` every
        instance of a family is byte-identical, so a row's "positive" would
        be the anchor's own bytes and the triplet objective would train on a
        degenerate pair. The producer must refuse, not emit it."""
        with self.assertRaises(ValueError):
            gfi.generate_corpus(rows=2, size=8, seed=1, jitter=0)


class CliTests(unittest.TestCase):
    def test_main_writes_the_expected_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "img"
            rc = gfi.main(
                [
                    "--rows", "5",
                    "--size", "16",
                    "--seed", "7",
                    "--out-dir", str(out),
                    "--families", "3",
                    "--instances-per-family", "2",
                ]
            )
            self.assertEqual(rc, 0)
            jsonl = out / "triplets.jsonl"
            self.assertEqual(len(jsonl.read_text().splitlines()), 5)
            pngs = sorted(p.name for p in out.glob("*.png"))
            self.assertEqual(len(pngs), 3 * 2)
            for name in pngs:
                w, h, _ = gfi.decode_png_rgb((out / name).read_bytes())
                self.assertEqual((w, h), (16, 16))

    def test_main_returns_nonzero_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = gfi.main(
                ["--rows", "0", "--size", "8", "--seed", "1", "--out-dir", str(Path(tmp) / "x")]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
