#!/usr/bin/env python3
"""`gen_fixed_length_audio_corpus.py`'s own suite (issue #421 W2b): the
LENGTH/RATE guarantee read back off the emitted WAV header, determinism
(family J), the emitted JSONL schema pinned by a literal field-name check
against `crates/jammi-bench/src/main.rs::MediaTripletRow`, the triplet
separation asserted MECHANICALLY (intra-family sample distance strictly
below inter-family), and the input-validation refusals.

Every duration/rate assertion goes through `read_wav` -- i.e. through the
BYTES actually written -- never through the sample array the encoder was
handed, so a clip whose header disagreed with its payload could not pass.

Stdlib-only (`unittest` + `wave`), no network, no numpy, no soundfile.

Run: `python3 -m pytest ci/scripts/perf/test_gen_fixed_length_audio_corpus.py`
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_fixed_length_audio_corpus as gfa  # noqa: E402

# The exact keys `crates/jammi-bench/src/main.rs::MediaTripletRow`
# deserializes -- the SAME six the image producer emits, so one Rust row
# struct serves both media tasks. Pinned as a literal set.
_EXPECTED_KEYS = {
    "anchor_id",
    "anchor_path",
    "positive_id",
    "positive_path",
    "negative_id",
    "negative_path",
}


def _mean_abs_diff(a, b) -> float:
    if len(a) != len(b):
        raise AssertionError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


class LengthTests(unittest.TestCase):
    def test_every_clip_carries_exactly_the_declared_frame_count(self):
        for seconds, rate in ((0.5, 16000), (1.0, 8000), (0.25, 48000)):
            expected = gfa.frame_count(seconds, rate)
            files, _rows = gfa.generate_corpus(
                rows=3, seconds=seconds, sample_rate=rate, seed=5
            )
            self.assertTrue(files)
            for name, data in files.items():
                channels, width, got_rate, samples = gfa.read_wav(data)
                self.assertEqual(channels, 1, name)
                self.assertEqual(width, 2, name)
                self.assertEqual(got_rate, rate, name)
                self.assertEqual(
                    len(samples), expected,
                    f"{name}: {len(samples)} frames, expected {expected}",
                )

    def test_frame_count_is_the_single_definition(self):
        self.assertEqual(gfa.frame_count(0.5, 16000), 8000)
        self.assertEqual(gfa.frame_count(1.0, 44100), 44100)

    def test_samples_stay_inside_the_int16_range(self):
        files, _rows = gfa.generate_corpus(
            rows=2, seconds=0.1, sample_rate=16000, seed=3, jitter=30000
        )
        for name, data in files.items():
            _c, _w, _r, samples = gfa.read_wav(data)
            self.assertGreaterEqual(min(samples), -32768, name)
            self.assertLessEqual(max(samples), 32767, name)

    def test_read_wav_refuses_a_non_wav(self):
        with self.assertRaises(Exception):
            gfa.read_wav(b"not a riff wave file")


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_byte_identical_files_and_rows(self):
        a_files, a_rows = gfa.generate_corpus(rows=5, seconds=0.1, sample_rate=16000, seed=21)
        b_files, b_rows = gfa.generate_corpus(rows=5, seconds=0.1, sample_rate=16000, seed=21)
        self.assertEqual(a_files, b_files)
        self.assertEqual(a_rows, b_rows)

    def test_different_seed_changes_the_samples(self):
        a_files, _ = gfa.generate_corpus(rows=5, seconds=0.1, sample_rate=16000, seed=21)
        b_files, _ = gfa.generate_corpus(rows=5, seconds=0.1, sample_rate=16000, seed=22)
        self.assertEqual(set(a_files), set(b_files))
        self.assertNotEqual(a_files, b_files)

    def test_rows_count_does_not_perturb_the_audio_bytes(self):
        small_files, small_rows = gfa.generate_corpus(
            rows=3, seconds=0.05, sample_rate=16000, seed=4
        )
        large_files, large_rows = gfa.generate_corpus(
            rows=11, seconds=0.05, sample_rate=16000, seed=4
        )
        self.assertEqual(small_files, large_files)
        self.assertEqual(small_rows, large_rows[:3])

    def test_written_tree_is_byte_identical_across_two_writes(self):
        files, rows = gfa.generate_corpus(rows=4, seconds=0.05, sample_rate=16000, seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            gfa.write_corpus(files, rows, a, "triplets.jsonl")
            gfa.write_corpus(files, rows, b, "triplets.jsonl")
            a_names = sorted(p.name for p in a.iterdir())
            self.assertEqual(a_names, sorted(p.name for p in b.iterdir()))
            for name in a_names:
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)


class TripletStructureTests(unittest.TestCase):
    def test_intra_family_distance_is_strictly_below_inter_family(self):
        families, instances = 4, 3
        files, _rows = gfa.generate_corpus(
            rows=4,
            seconds=0.25,
            sample_rate=16000,
            seed=3,
            families=families,
            instances_per_family=instances,
        )
        samples = {name: gfa.read_wav(data)[3] for name, data in files.items()}

        intra, inter = [], []
        for f in range(families):
            for i in range(instances):
                for j in range(i + 1, instances):
                    intra.append(
                        _mean_abs_diff(samples[gfa._clip_name(f, i)], samples[gfa._clip_name(f, j)])
                    )
            for g in range(f + 1, families):
                for i in range(instances):
                    inter.append(
                        _mean_abs_diff(samples[gfa._clip_name(f, i)], samples[gfa._clip_name(g, i)])
                    )
        self.assertLess(
            max(intra),
            min(inter),
            f"every intra-family pair must be closer than every inter-family pair; "
            f"max intra={max(intra):.1f}, min inter={min(inter):.1f}",
        )

    def test_every_row_pairs_one_family_against_another(self):
        families, instances = 4, 4
        _files, rows = gfa.generate_corpus(
            rows=13,
            seconds=0.05,
            sample_rate=16000,
            seed=2,
            families=families,
            instances_per_family=instances,
        )

        def fam_of(path: str) -> int:
            return int(path.split("_f")[1].split("_i")[0])

        for row in rows:
            a = fam_of(row["anchor_path"])
            p = fam_of(row["positive_path"])
            n = fam_of(row["negative_path"])
            self.assertEqual(a, p, f"anchor/positive must share a family: {row}")
            self.assertNotEqual(a, n, f"negative must come from another family: {row}")
            self.assertNotEqual(row["anchor_path"], row["positive_path"], f"{row}")

    def test_two_instances_of_one_family_are_never_byte_identical(self):
        files, _rows = gfa.generate_corpus(
            rows=2, seconds=0.05, sample_rate=16000, seed=6, instances_per_family=2
        )
        self.assertNotEqual(files[gfa._clip_name(0, 0)], files[gfa._clip_name(0, 1)])


class SchemaTests(unittest.TestCase):
    def test_every_row_has_exactly_the_six_expected_keys(self):
        _files, rows = gfa.generate_corpus(rows=7, seconds=0.05, sample_rate=16000, seed=1)
        for row in rows:
            self.assertEqual(set(row.keys()), _EXPECTED_KEYS)
            for key in _EXPECTED_KEYS:
                self.assertIsInstance(row[key], str)
                self.assertTrue(row[key])

    def test_every_referenced_path_exists_and_is_relative_to_the_jsonl(self):
        files, rows = gfa.generate_corpus(rows=9, seconds=0.05, sample_rate=16000, seed=12)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "corpus"
            jsonl = gfa.write_corpus(files, rows, out, "triplets.jsonl")
            for line in jsonl.read_text().splitlines():
                row = json.loads(line)
                for key in ("anchor_path", "positive_path", "negative_path"):
                    rel = Path(row[key])
                    self.assertFalse(rel.is_absolute(), f"{key} must be relative: {row[key]!r}")
                    self.assertTrue((jsonl.parent / rel).is_file(), f"missing {row[key]}")

    def test_ids_are_unique_across_rows_and_roles(self):
        _files, rows = gfa.generate_corpus(rows=15, seconds=0.05, sample_rate=16000, seed=9)
        ids = []
        for row in rows:
            ids.extend([row["anchor_id"], row["positive_id"], row["negative_id"]])
        self.assertEqual(len(ids), len(set(ids)))

    def test_image_and_audio_producers_agree_on_the_row_schema(self):
        """One Rust row struct serves both media tasks, so the two producers
        must emit the SAME key set -- checked against the other producer's
        own output, not merely against this file's literal."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import gen_fixed_shape_image_corpus as gfi  # noqa: PLC0415

        _f, img_rows = gfi.generate_corpus(rows=2, size=8, seed=1)
        _g, aud_rows = gfa.generate_corpus(rows=2, seconds=0.05, sample_rate=16000, seed=1)
        self.assertEqual(set(img_rows[0].keys()), set(aud_rows[0].keys()))
        self.assertEqual(set(aud_rows[0].keys()), _EXPECTED_KEYS)


class ValidationTests(unittest.TestCase):
    def test_nonpositive_rows_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=0, seconds=0.05, sample_rate=16000, seed=1)

    def test_nonpositive_seconds_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=2, seconds=0.0, sample_rate=16000, seed=1)

    def test_nonpositive_sample_rate_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=2, seconds=0.05, sample_rate=0, seed=1)

    def test_a_duration_that_rounds_to_zero_frames_is_refused(self):
        """Non-vacuous domain edge: `--seconds 1e-9` is positive and
        `--sample-rate` is positive, yet their product rounds to zero
        frames. An empty clip would sail through both scalar checks and
        produce a WAV the audio front end cannot fuse."""
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=2, seconds=1e-9, sample_rate=16000, seed=1)

    def test_one_family_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=2, seconds=0.05, sample_rate=16000, seed=1, families=1)

    def test_one_instance_per_family_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(
                rows=2, seconds=0.05, sample_rate=16000, seed=1, instances_per_family=1
            )

    def test_zero_jitter_refused(self):
        with self.assertRaises(ValueError):
            gfa.generate_corpus(rows=2, seconds=0.05, sample_rate=16000, seed=1, jitter=0)


class CliTests(unittest.TestCase):
    def test_main_writes_the_expected_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aud"
            rc = gfa.main(
                [
                    "--rows", "5",
                    "--seconds", "0.05",
                    "--sample-rate", "16000",
                    "--seed", "7",
                    "--out-dir", str(out),
                    "--families", "3",
                    "--instances-per-family", "2",
                ]
            )
            self.assertEqual(rc, 0)
            jsonl = out / "triplets.jsonl"
            self.assertEqual(len(jsonl.read_text().splitlines()), 5)
            wavs = sorted(p.name for p in out.glob("*.wav"))
            self.assertEqual(len(wavs), 3 * 2)
            for name in wavs:
                _c, _w, rate, samples = gfa.read_wav((out / name).read_bytes())
                self.assertEqual(rate, 16000)
                self.assertEqual(len(samples), gfa.frame_count(0.05, 16000))

    def test_main_returns_nonzero_on_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = gfa.main(
                [
                    "--rows", "0",
                    "--seconds", "0.05",
                    "--sample-rate", "16000",
                    "--seed", "1",
                    "--out-dir", str(Path(tmp) / "x"),
                ]
            )
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
