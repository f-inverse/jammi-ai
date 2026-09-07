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


class PoolCacheTests(unittest.TestCase):
    """`--pool-cache-dir` (esc-088 round-4 advisory: hermetic dry-run suite
    runtime): opt-in, real runs never set it. Every assertion here drives
    the REAL CLI (`gfi.main`), never `_build_pool`/`_load_or_build_pool`
    directly, so a cache-path bug in argument plumbing cannot hide behind
    a unit-level call that bypasses it."""

    @staticmethod
    def _sha256_tree(d: Path) -> dict[str, str]:
        import hashlib

        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir())
            if p.is_file()
        }

    def test_cache_hit_and_miss_are_byte_identical_to_the_uncached_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            uncached = Path(tmp) / "uncached"
            cache_dir = Path(tmp) / "cache"
            miss_out = Path(tmp) / "miss"
            hit_out = Path(tmp) / "hit"
            common = [
                "--rows", "5", "--size", "12", "--seed", "3",
                "--families", "4", "--instances-per-family", "3",
            ]
            self.assertEqual(gfi.main([*common, "--out-dir", str(uncached)]), 0)
            # First cached call: a cache MISS (builds + populates the cache).
            self.assertEqual(
                gfi.main([*common, "--out-dir", str(miss_out), "--pool-cache-dir", str(cache_dir)]),
                0,
            )
            # Second cached call, a DIFFERENT --rows (never part of the pool
            # key) and --out-dir: a cache HIT (reads the pool off disk).
            self.assertEqual(
                gfi.main(
                    [
                        "--rows", "2", "--size", "12", "--seed", "3",
                        "--families", "4", "--instances-per-family", "3",
                        "--out-dir", str(hit_out), "--pool-cache-dir", str(cache_dir),
                    ]
                ),
                0,
            )
            uncached_pngs = self._sha256_tree(uncached)
            miss_pngs = {k: v for k, v in self._sha256_tree(miss_out).items() if k.endswith(".png")}
            hit_pngs = {k: v for k, v in self._sha256_tree(hit_out).items() if k.endswith(".png")}
            uncached_only_pngs = {k: v for k, v in uncached_pngs.items() if k.endswith(".png")}
            self.assertTrue(uncached_only_pngs, "no PNGs found -- test is vacuous")
            self.assertEqual(uncached_only_pngs, miss_pngs)
            self.assertEqual(uncached_only_pngs, hit_pngs)

    def test_a_different_pool_shape_gets_a_different_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            gfi.main([
                "--rows", "4", "--size", "10", "--seed", "1", "--families", "2",
                "--instances-per-family", "2", "--out-dir", str(Path(tmp) / "a"),
                "--pool-cache-dir", str(cache_dir),
            ])
            gfi.main([
                "--rows", "4", "--size", "10", "--seed", "2", "--families", "2",
                "--instances-per-family", "2", "--out-dir", str(Path(tmp) / "b"),
                "--pool-cache-dir", str(cache_dir),
            ])
            subdirs = [p for p in cache_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(subdirs), 2, "two different seeds must land in two different cache keys")

    def test_cache_dir_left_unset_never_touches_the_filesystem_beyond_out_dir(self):
        """The default (no `--pool-cache-dir`, every real invocation): the
        producer must not silently create or read any cache directory."""
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).iterdir())
            gfi.main([
                "--rows", "3", "--size", "8", "--seed", "1", "--out-dir", str(Path(tmp) / "out"),
            ])
            after = set(Path(tmp).iterdir())
            self.assertEqual(after - before, {Path(tmp) / "out"})


class HeldOutSplitTests(unittest.TestCase):
    """`--heldout-rows` (issue #421 P1-b(iv)): the held-out split's ROW
    COUNT, its FAMILY-disjointness from the train split, its determinism,
    and every refusal the family pool / batch divisibility can produce.

    Disjointness is asserted on the FILE SETS the two row lists actually
    reference -- not on the family indices the producer used internally --
    so the assertion holds against what a consumer reads, and a future
    change to the family-window arithmetic that reintroduced an overlap
    reds here even if the internal bookkeeping still looked right.
    """

    @staticmethod
    def _files_referenced(rows):
        out = set()
        for row in rows:
            out.update({row["anchor_path"], row["positive_path"], row["negative_path"]})
        return out

    @staticmethod
    def _ids(rows):
        out = set()
        for row in rows:
            out.update({row["anchor_id"], row["positive_id"], row["negative_id"]})
        return out

    def test_row_count_is_exactly_what_was_asked_for(self):
        _files, train, heldout = gfi.generate_split(
            rows=12, size=8, seed=3, families=6, heldout_rows=4, heldout_batch=2
        )
        self.assertEqual(len(train), 12)
        self.assertEqual(len(heldout), 4)

    def test_the_two_splits_reference_disjoint_files_and_disjoint_ids(self):
        _files, train, heldout = gfi.generate_split(
            rows=12, size=8, seed=3, families=6, heldout_rows=4, heldout_batch=2
        )
        train_files = self._files_referenced(train)
        heldout_files = self._files_referenced(heldout)
        self.assertTrue(train_files, "the train split must reference some files at all")
        self.assertTrue(heldout_files, "the held-out split must reference some files at all")
        self.assertEqual(
            train_files & heldout_files,
            set(),
            "a family-disjoint split must share NO image file between the two halves",
        )
        self.assertEqual(
            self._ids(train) & self._ids(heldout),
            set(),
            "the two splits' id spaces must be disjoint (they are joined BY id downstream)",
        )

    def test_every_referenced_file_actually_exists_in_the_emitted_corpus(self):
        """The held-out rows name files from the RESERVED families, which
        the producer still generates -- a split that referenced a file it
        never wrote would fail at load time on the pod, not here."""
        files, train, heldout = gfi.generate_split(
            rows=12, size=8, seed=3, families=6, heldout_rows=4, heldout_batch=2
        )
        for name in self._files_referenced(train) | self._files_referenced(heldout):
            self.assertIn(name, files, f"{name} is referenced but was never emitted")

    def test_heldout_rows_carry_the_train_schema_and_the_pinned_shape(self):
        files, _train, heldout = gfi.generate_split(
            rows=8, size=16, seed=4, families=6, heldout_rows=4, heldout_batch=4
        )
        for row in heldout:
            self.assertEqual(set(row.keys()), _EXPECTED_KEYS)
            for key in ("anchor_path", "positive_path", "negative_path"):
                w, h, _pixels = gfi.decode_png_rgb(files[row[key]])
                self.assertEqual((w, h), (16, 16))

    def test_a_heldout_row_is_a_real_triplet_same_family_positive(self):
        """Non-vacuity: the held-out rows must be well-formed TRIPLETS in
        their own right (anchor and positive from ONE family, negative from
        a different one) -- a "disjoint" split of degenerate rows would
        satisfy every disjointness assertion above and be useless."""
        _files, _train, heldout = gfi.generate_split(
            rows=8, size=8, seed=4, families=6, heldout_rows=4, heldout_batch=4
        )
        for row in heldout:
            anchor_fam = row["anchor_path"].split("_")[1]
            positive_fam = row["positive_path"].split("_")[1]
            negative_fam = row["negative_path"].split("_")[1]
            self.assertEqual(anchor_fam, positive_fam, row)
            self.assertNotEqual(anchor_fam, negative_fam, row)

    def test_the_train_split_is_byte_identical_to_a_run_without_a_heldout_request(self):
        """Adding a held-out split must not perturb the train corpus's IMAGE
        bytes; the train ROW list legitimately narrows (it now draws from
        the unreserved families only), which is the whole point of the
        reservation -- so this pins the two claims separately rather than
        conflating them."""
        files_plain, train_plain = gfi.generate_corpus(rows=12, size=8, seed=3, families=6)
        files_split, train_split, _heldout = gfi.generate_split(
            rows=12, size=8, seed=3, families=6, heldout_rows=4, heldout_batch=2
        )
        self.assertEqual(files_plain, files_split, "the emitted image bytes must not change")
        self.assertNotEqual(
            train_plain,
            train_split,
            "the train rows MUST narrow to the unreserved families -- identical row lists would "
            "mean the reservation did nothing and the split is not actually disjoint",
        )

    def test_same_seed_same_bytes(self):
        a = gfi.generate_split(rows=8, size=8, seed=11, families=6, heldout_rows=4, heldout_batch=2)
        b = gfi.generate_split(rows=8, size=8, seed=11, families=6, heldout_rows=4, heldout_batch=2)
        self.assertEqual(a, b)

    def test_refuses_when_the_family_pool_cannot_support_the_split(self):
        # 3 families, 2 reserved -> 1 left for train, below the 2 a triplet
        # needs. A REFUSAL, never a silently overlapping split.
        with self.assertRaises(ValueError) as ctx:
            gfi.generate_split(
                rows=4, size=8, seed=1, families=3, heldout_rows=2, heldout_batch=2
            )
        self.assertIn("--families", str(ctx.exception))
        # The default pool (4) reserving 2 leaves exactly 2 -- the boundary
        # case must be ACCEPTED, so the refusal above is a real bound and
        # not an off-by-one that rejects every legal split.
        _files, train, heldout = gfi.generate_split(
            rows=4, size=8, seed=1, families=4, heldout_rows=2, heldout_batch=2
        )
        self.assertEqual((len(train), len(heldout)), (4, 2))

    def test_refuses_a_heldout_family_count_below_two(self):
        with self.assertRaises(ValueError) as ctx:
            gfi.generate_split(
                rows=4, size=8, seed=1, families=6, heldout_rows=2, heldout_batch=2,
                heldout_families=1,
            )
        self.assertIn("--heldout-families", str(ctx.exception))

    def test_refuses_a_heldout_row_count_that_is_not_a_multiple_of_the_batch(self):
        with self.assertRaises(ValueError) as ctx:
            gfi.generate_split(
                rows=8, size=8, seed=1, families=6, heldout_rows=5, heldout_batch=2
            )
        self.assertIn("--heldout-batch", str(ctx.exception))

    def test_refuses_a_heldout_request_with_no_batch_stated(self):
        with self.assertRaises(ValueError) as ctx:
            gfi.generate_split(rows=8, size=8, seed=1, families=6, heldout_rows=4)
        self.assertIn("--heldout-batch is required", str(ctx.exception))

    def test_cli_writes_both_heldout_files_and_they_agree_row_for_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "img"
            rc = gfi.main([
                "--rows", "8",
                "--size", "8",
                "--seed", "5",
                "--out-dir", str(out),
                "--families", "6",
                "--heldout-rows", "4",
                "--heldout-batch", "2",
            ])
            self.assertEqual(rc, 0)
            ids_lines = (out / "heldout_ids.txt").read_text().splitlines()
            jsonl_lines = (out / "heldout_triplets.jsonl").read_text().splitlines()
            self.assertEqual(len(ids_lines), 4)
            self.assertEqual(len(jsonl_lines), 4)
            for ids_line, jsonl_line in zip(ids_lines, jsonl_lines, strict=True):
                anchor, positive, negative = ids_line.split("\t")
                row = json.loads(jsonl_line)
                # The ids file IS the scoring order and the JSONL is joined
                # to it BY anchor_id: a mismatch here is a fixture the
                # loader would refuse (or, worse, silently reorder).
                self.assertEqual(
                    (anchor, positive, negative),
                    (row["anchor_id"], row["positive_id"], row["negative_id"]),
                )
            # And the same run WITHOUT the flag writes neither file.
            plain = Path(tmp) / "img_plain"
            self.assertEqual(
                gfi.main([
                    "--rows", "8", "--size", "8", "--seed", "5",
                    "--out-dir", str(plain), "--families", "6",
                ]),
                0,
            )
            self.assertFalse((plain / "heldout_ids.txt").exists())
            self.assertFalse((plain / "heldout_triplets.jsonl").exists())

    def test_cli_bytes_are_deterministic_across_two_runs(self):
        """Family J, at the FILE level: the same argv twice must write
        byte-identical trees, held-out files included."""
        def run(dirpath):
            self.assertEqual(
                gfi.main([
                    "--rows", "8", "--size", "8", "--seed", "9",
                    "--out-dir", str(dirpath), "--families", "6",
                    "--heldout-rows", "4", "--heldout-batch", "4",
                ]),
                0,
            )
            return {p.name: p.read_bytes() for p in sorted(dirpath.iterdir())}

        with tempfile.TemporaryDirectory() as tmp:
            first = run(Path(tmp) / "a")
            second = run(Path(tmp) / "b")
            self.assertEqual(first.keys(), second.keys())
            self.assertEqual(first, second)
            self.assertIn("heldout_ids.txt", first)
            self.assertIn("heldout_triplets.jsonl", first)

    def test_cli_returns_nonzero_on_an_unsupportable_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = gfi.main([
                "--rows", "4", "--size", "8", "--seed", "1",
                "--out-dir", str(Path(tmp) / "x"), "--families", "3",
                "--heldout-rows", "2", "--heldout-batch", "2",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse(
                (Path(tmp) / "x").exists(),
                "a refused split must write NOTHING -- not a partial corpus",
            )



if __name__ == "__main__":
    unittest.main()
