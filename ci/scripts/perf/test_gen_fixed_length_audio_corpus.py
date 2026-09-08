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

import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
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


def _load_variant_module(replacements: list[tuple[str, str]], tag: str, add_cleanup) -> object:
    """Import a temp copy of `gen_fixed_length_audio_corpus.py` with each
    `(old, new)` in `replacements` applied as a literal one-occurrence
    source substitution -- a REAL file on disk, imported under a FRESH
    module name. `_pool_cache_key` hashes `Path(__file__).read_bytes()`,
    so the only way to prove a real source edit moves the key is to give
    the key function an actual, differently-byted file to hash; nothing
    short of a real edited file on disk exercises that read path.
    `add_cleanup` is normally a `TestCase.addCleanup` bound method, so the
    temp directory is removed once the calling test finishes."""
    src = Path(gfa.__file__).read_text(encoding="utf-8")
    for old, new in replacements:
        occurrences = src.count(old)
        if occurrences != 1:
            raise AssertionError(f"expected exactly one occurrence of {old!r}, found {occurrences}")
        src = src.replace(old, new, 1)
    tmp_dir = tempfile.mkdtemp(prefix=f"gfa_variant_{tag}_")
    add_cleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
    mod_path = Path(tmp_dir) / f"gfa_variant_{tag}.py"
    mod_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


class PoolCacheTests(unittest.TestCase):
    """`--pool-cache-dir` (esc-088 round-4 advisory: hermetic dry-run suite
    runtime): opt-in, real runs never set it. Every assertion here drives
    the REAL CLI (`gfa.main`), never `_build_pool`/`_load_or_build_pool`
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
        """Drives the PRODUCTION shape
        (`--heldout-rows`/`--heldout-families`, exactly as
        `profile_421_legs.sh` calls this producer) and compares EVERY
        emitted file, not just the WAVs -- `triplets.jsonl`,
        `heldout_ids.txt`, and `heldout_triplets.jsonl` all move under a
        `--rows`/held-out-shape change the same way the audio pool moves
        under a `--families`/`--seed` change, and a parity oracle that only
        ever looked at `.wav` names could not catch a regression in any of
        the three.

        `uncached` and `miss` share the EXACT SAME arguments (including
        `--rows`), so every file they emit -- pool WAVs AND
        rows/held-out-derived JSONL/txt -- must be byte-for-byte identical.
        `hit` deliberately varies `--rows` (never part of the pool cache
        key -- see `_pool_cache_key`), so only its WAVs are compared; its
        JSONL/held-out files differ from the other two BY DESIGN, proving
        `--rows` reaches the row/held-out output without disturbing the
        cached pool."""
        with tempfile.TemporaryDirectory() as tmp:
            uncached = Path(tmp) / "uncached"
            cache_dir = Path(tmp) / "cache"
            miss_out = Path(tmp) / "miss"
            hit_out = Path(tmp) / "hit"
            common = [
                "--rows", "8", "--seconds", "0.05", "--sample-rate", "16000", "--seed", "3",
                "--families", "6", "--instances-per-family", "3",
                "--heldout-rows", "4", "--heldout-batch", "2", "--heldout-families", "2",
            ]
            self.assertEqual(gfa.main([*common, "--out-dir", str(uncached)]), 0)
            # First cached call: a cache MISS (builds + populates the cache).
            self.assertEqual(
                gfa.main([*common, "--out-dir", str(miss_out), "--pool-cache-dir", str(cache_dir)]),
                0,
            )
            # Second cached call, a DIFFERENT --rows (never part of the pool
            # key) and --out-dir: a cache HIT (reads the pool off disk).
            self.assertEqual(
                gfa.main(
                    [
                        "--rows", "2", "--seconds", "0.05", "--sample-rate", "16000",
                        "--seed", "3", "--families", "6", "--instances-per-family", "3",
                        "--heldout-rows", "2", "--heldout-batch", "2", "--heldout-families", "2",
                        "--out-dir", str(hit_out), "--pool-cache-dir", str(cache_dir),
                    ]
                ),
                0,
            )
            # Non-vacuousness control: `hit`'s call above must actually have
            # taken the HIT branch (read the pool `_build_pool` already
            # wrote for `miss`), not silently rebuilt it under a second
            # cache key -- nothing else in this test would distinguish a
            # cache-key bug (`hit` never matching `miss`'s key, and so
            # rebuilding from scratch every time) from a real hit, since the
            # WAV bytes would come out identical either way (same seed/
            # shape). `miss` and `hit` share the exact same pool-shaping
            # arguments, so exactly ONE cache-key subdirectory must exist
            # after both calls.
            subdirs_after_hit = [p for p in cache_dir.iterdir() if p.is_dir()]
            self.assertEqual(
                len(subdirs_after_hit), 1, "the hit call must reuse the miss call's cache key, not create a second one"
            )
            uncached_all = self._sha256_tree(uncached)
            miss_all = self._sha256_tree(miss_out)
            hit_all = self._sha256_tree(hit_out)
            expected_names = {
                "triplets.jsonl", "heldout_ids.txt", "heldout_triplets.jsonl",
            }
            self.assertTrue(expected_names.issubset(uncached_all), uncached_all.keys())
            self.assertTrue(any(k.endswith(".wav") for k in uncached_all), "no WAVs -- test is vacuous")
            # Identical arguments -> EVERY emitted file byte-identical.
            self.assertEqual(uncached_all, miss_all)
            # Different --rows/held-out-rows -> pool WAVs still identical...
            uncached_wavs = {k: v for k, v in uncached_all.items() if k.endswith(".wav")}
            hit_wavs = {k: v for k, v in hit_all.items() if k.endswith(".wav")}
            self.assertEqual(uncached_wavs, hit_wavs)
            # ...but the row/held-out-derived files differ BY DESIGN (a
            # different --rows/--heldout-rows genuinely changes their
            # content) -- asserting the difference keeps this a real
            # non-vacuous control, not an accidental byte-identity that
            # would mask a cache bug leaking downstream output.
            for name in expected_names:
                self.assertNotEqual(uncached_all[name], hit_all[name], name)

    def test_a_different_pool_shape_gets_a_different_cache_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            gfa.main([
                "--rows", "4", "--seconds", "0.05", "--sample-rate", "16000", "--seed", "1",
                "--families", "2", "--instances-per-family", "2",
                "--out-dir", str(Path(tmp) / "a"), "--pool-cache-dir", str(cache_dir),
            ])
            gfa.main([
                "--rows", "4", "--seconds", "0.05", "--sample-rate", "16000", "--seed", "2",
                "--families", "2", "--instances-per-family", "2",
                "--out-dir", str(Path(tmp) / "b"), "--pool-cache-dir", str(cache_dir),
            ])
            subdirs = [p for p in cache_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(subdirs), 2, "two different seeds must land in two different cache keys")

    def test_cache_dir_left_unset_never_touches_the_filesystem_beyond_out_dir(self):
        """The default (no `--pool-cache-dir`, every real invocation): the
        producer must not silently create or read any cache directory."""
        with tempfile.TemporaryDirectory() as tmp:
            before = set(Path(tmp).iterdir())
            gfa.main([
                "--rows", "3", "--seconds", "0.05", "--sample-rate", "16000", "--seed", "1",
                "--out-dir", str(Path(tmp) / "out"),
            ])
            after = set(Path(tmp).iterdir())
            self.assertEqual(after - before, {Path(tmp) / "out"})

    def test_unchanged_construction_code_hits_the_same_cache_key(self):
        """Calling the key function twice with nothing edited must be
        IDENTICAL: the fingerprint is a pure function of its arguments and
        this file's current on-disk bytes, so two calls against unchanged
        source can never disagree."""
        args = (4, 3, 800, 16000, 200, 1)
        self.assertEqual(gfa._pool_cache_key(*args), gfa._pool_cache_key(*args))

    def test_a_marker_recorded_for_a_different_shape_under_the_same_key_is_refused(self):
        """Belt-and-braces at the point of use (round-5 audit B1(b)): a
        `_DONE` marker's OWN recorded shape must agree, BY NAME, with the
        shape this call actually requested -- checked even though `key`
        already claims to identify the requested shape uniquely. Simulates
        the one way that claim could fail (a hand-corrupted marker, or a
        would-be cache-key collision) by writing a marker for a DIFFERENT
        `frames` value into the exact directory this key resolves to, then
        driving the real lookup path (`_load_or_build_pool`, never the
        marker file directly) at the ORIGINAL shape."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            args = (4, 3, 800, 16000, 200, 1)  # families, instances, frames, sample_rate, jitter, seed
            key = gfa._pool_cache_key(*args)
            key_dir = cache_dir / key
            key_dir.mkdir(parents=True)
            (key_dir / gfa._POOL_CACHE_DONE_MARKER).write_text(
                "families=4 instances_per_family=3 frames=799 sample_rate=16000 jitter=200 seed=1\n"
            )
            with self.assertRaises(ValueError) as ctx:
                gfa._load_or_build_pool(cache_dir, *args)
            self.assertIn("799", str(ctx.exception))
            self.assertIn("800", str(ctx.exception))

    def test_a_marker_recorded_with_a_missing_field_is_refused(self):
        """A marker missing a WHOLE determinant field entirely (never
        written, not merely wrong-valued) is refused the same way a
        wrong-valued one is: `_parse_pool_marker` happily parses a
        partial marker (it does not know which fields are "required"),
        but the resulting dict then compares unequal (fewer keys) to the
        fully-populated `requested` shape, so `_load_or_build_pool`'s
        belt-and-braces check still refuses."""
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            args = (4, 3, 800, 16000, 200, 1)
            key = gfa._pool_cache_key(*args)
            key_dir = cache_dir / key
            key_dir.mkdir(parents=True)
            (key_dir / gfa._POOL_CACHE_DONE_MARKER).write_text(
                "families=4 instances_per_family=3 frames=800 sample_rate=16000 seed=1\n"  # jitter missing
            )
            with self.assertRaises(ValueError):
                gfa._load_or_build_pool(cache_dir, *args)

    def test_marker_round_trips_through_write_and_parse(self):
        """[`gfa._pool_marker_text`]/[`gfa._parse_pool_marker`] are exact
        inverses for every field the marker carries -- the single
        definition the write path (a cache MISS) and the read path (a
        cache HIT's belt-and-braces check) both go through, so they cannot
        independently drift on the field set."""
        args = dict(families=6, instances_per_family=3, frames=4800, sample_rate=48000, jitter=50, seed=9)
        text = gfa._pool_marker_text(**args)
        parsed = gfa._parse_pool_marker(text)
        self.assertEqual(parsed, args)


class ParsePoolMarkerTests(unittest.TestCase):
    """[`gfa._parse_pool_marker`]'s own error-handling arms -- malformed,
    duplicate, and non-int tokens each refuse rather than guess, and an
    empty marker parses to an empty dict rather than raising (the
    documented, intentional "nothing recorded" shape a caller-level
    dict-equality check against a fully-populated `requested` shape then
    catches as a mismatch, exercised separately by
    `PoolCacheTests.test_a_marker_recorded_with_a_missing_field_is_refused`
    above)."""

    def test_malformed_token_without_equals_raises(self):
        with self.assertRaises(ValueError) as ctx:
            gfa._parse_pool_marker("families=4 bogus")
        self.assertIn("bogus", str(ctx.exception))

    def test_duplicate_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            gfa._parse_pool_marker("families=4 families=5")
        self.assertIn("families=5", str(ctx.exception))

    def test_non_int_value_raises(self):
        with self.assertRaises(ValueError):
            gfa._parse_pool_marker("families=abc")

    def test_empty_text_parses_to_empty_dict(self):
        self.assertEqual(gfa._parse_pool_marker(""), {})


class PoolCacheKeyDeterminantTests(unittest.TestCase):
    """The determinant SET is generated from ONE source,
    `gfa.POOL_KEY_DETERMINANTS`, never hand-copied into this test: it is
    asserted equal to the set of keyword arguments `_pool_cache_key`
    actually accepts (introspected via `inspect.signature`), and the loop
    below iterates that SAME tuple -- so a determinant added to the
    production tuple (and the function's signature) later is picked up by
    this test automatically, never left untested until someone remembers
    to update a parallel hand-written list here."""

    _BASE = dict(families=4, instances_per_family=3, frames=800, sample_rate=16000, jitter=200, seed=1)

    def test_determinant_tuple_matches_the_pool_cache_key_signature(self):
        sig_params = set(inspect.signature(gfa._pool_cache_key).parameters)
        self.assertEqual(set(gfa.POOL_KEY_DETERMINANTS), sig_params)

    def test_build_pool_signature_matches_the_pool_cache_key_signature(self):
        """`POOL_KEY_DETERMINANTS` is pinned against `_pool_cache_key`'s own
        signature above, but the determinant SET only actually closes the
        class if `_build_pool` -- the function whose OUTPUT the cache key is
        supposed to determine -- cannot itself accept an argument the key
        does not cover. Asserted directly off both signatures (never a
        hand-copied parameter list) so a future `_build_pool` parameter
        added without a matching `_pool_cache_key` parameter (or vice versa)
        fails HERE, not as a silent stale-cache-hit bug on disk."""
        build_pool_params = set(inspect.signature(gfa._build_pool).parameters)
        cache_key_params = set(inspect.signature(gfa._pool_cache_key).parameters)
        self.assertEqual(
            build_pool_params,
            cache_key_params,
            "_build_pool's parameters must equal _pool_cache_key's (every output-affecting "
            "determinant of _build_pool must be in the cache key, and the cache key must name "
            "no determinant _build_pool does not actually consume)",
        )

    @staticmethod
    def _key(**overrides) -> str:
        kw = {**PoolCacheKeyDeterminantTests._BASE, **overrides}
        return gfa._pool_cache_key(**kw)

    def test_each_determinant_moves_the_key_alone(self):
        baseline = self._key()
        for determinant in gfa.POOL_KEY_DETERMINANTS:
            with self.subTest(determinant=determinant):
                bumped = self._key(**{determinant: self._BASE[determinant] + 1})
                self.assertNotEqual(
                    baseline, bumped, f"changing {determinant!r} alone must move the cache key"
                )

    def test_interpreter_version_is_a_determinant(self):
        """`sys.version_info[:2]` is folded into the canonical string
        directly off the real `sys` module (never an argument), so this
        one determinant is exercised by patching the ACTUAL global state
        `_pool_cache_key` reads -- not a source edit -- since there is no
        other way to observe a second interpreter version in one process."""
        baseline = self._key()
        with unittest.mock.patch.object(gfa.sys, "version_info", (3, 1, 0, "final", 0)):
            other = self._key()
        self.assertNotEqual(baseline, other)

    def test_marker_fields_equal_the_determinant_tuple(self):
        """`_pool_marker_text`/`_parse_pool_marker` carry exactly the
        `POOL_KEY_DETERMINANTS` field set -- generated from the same
        tuple `_pool_cache_key` iterates, so the two can never drift on
        which fields they cover."""
        text = gfa._pool_marker_text(**self._BASE)
        parsed = gfa._parse_pool_marker(text)
        self.assertEqual(set(parsed), set(gfa.POOL_KEY_DETERMINANTS))
        self.assertEqual(parsed, self._BASE)


class PoolKeyDeterminantAddMutantTests(unittest.TestCase):
    """An ADD-mutant proof that the determinant loop covers a NEW
    determinant automatically once it is wired into all three of
    `_pool_cache_key`'s signature, its own `_pool_key_values` call, and
    `POOL_KEY_DETERMINANTS` -- and that wiring it into the first two
    WITHOUT the third is refused at key-build time (`_pool_key_values`'s
    own have/want set-equality check), never silently ignored."""

    _SIG_OLD = "    seed: int,\n) -> str:"
    _SIG_NEW = "    seed: int,\n    channels: int,\n) -> str:"
    # Anchored on the text immediately AFTER the closing `)` (`_pool_cache_
    # key`'s own next line) so this substitution targets ONLY its
    # `_pool_key_values(...)` call, not `_pool_marker_text`'s identically
    # worded one a few lines below.
    _CALL_OLD = "        jitter=jitter,\n        seed=seed,\n    )\n    module_bytes"
    _CALL_NEW = "        jitter=jitter,\n        seed=seed,\n        channels=channels,\n    )\n    module_bytes"
    _TUPLE_OLD = '    "jitter",\n    "seed",\n)'
    _TUPLE_NEW = '    "jitter",\n    "seed",\n    "channels",\n)'

    def test_new_determinant_wired_into_signature_call_and_tuple_moves_the_key(self):
        variant = _load_variant_module(
            [(self._SIG_OLD, self._SIG_NEW), (self._CALL_OLD, self._CALL_NEW), (self._TUPLE_OLD, self._TUPLE_NEW)],
            "add_channels_wired",
            self.addCleanup,
        )
        base = dict(families=4, instances_per_family=3, frames=800, sample_rate=16000, jitter=200, seed=1)
        key_a = variant._pool_cache_key(**base, channels=1)
        key_b = variant._pool_cache_key(**base, channels=2)
        self.assertNotEqual(key_a, key_b, "a determinant wired into signature+call+tuple must move the key alone")

    def test_new_determinant_wired_into_signature_and_call_but_not_tuple_is_refused(self):
        variant = _load_variant_module(
            [(self._SIG_OLD, self._SIG_NEW), (self._CALL_OLD, self._CALL_NEW)],
            "add_channels_unwired",
            self.addCleanup,
        )
        base = dict(families=4, instances_per_family=3, frames=800, sample_rate=16000, jitter=200, seed=1)
        with self.assertRaises(KeyError):
            variant._pool_cache_key(**base, channels=1)


class RealSourceEditMovesTheKeyTests(unittest.TestCase):
    """`_pool_cache_key` hashes the producer MODULE'S OWN SOURCE BYTES
    (`Path(__file__).read_bytes()`), so the key is intentionally MAXIMAL: a
    real on-disk edit to ANY line of this module -- a function the pool
    construction actually calls, a constant, a function the pool
    construction never calls (`read_wav`), or even the module docstring --
    must move the key. No false cache hit is possible under this design:
    two module states that differ by even one byte always hash to two
    different keys. The cost of that completeness is a comment-only edit
    forcing one extra pool regeneration -- seconds, in a test-only cache --
    which is why every case below is a POSITIVE control (the key moves),
    not a mix of positive and negative ones.

    These tests take the PRODUCER'S OWN SOURCE FILE, apply exactly one
    textual edit to a single line (each substring asserted unique in the
    file before the edit), import the edited copy under a FRESH module
    name via `_load_variant_module`, and compare its `_pool_cache_key`
    output against the unedited module's for identical arguments -- a real
    code edit, not an object swap, so a change that only a monkeypatch
    could observe (and this key deliberately does not need to observe)
    is never mistaken for proof."""

    _ARGS = (4, 3, 800, 16000, 200, 1)  # families, instances_per_family, frames, sample_rate, jitter, seed

    def setUp(self) -> None:
        self.baseline = gfa._pool_cache_key(*self._ARGS)

    def _key_after(self, old: str, new: str, tag: str) -> str:
        variant = _load_variant_module([(old, new)], tag, self.addCleanup)
        return variant._pool_cache_key(*self._ARGS)

    def test_editing_build_pool_moves_the_key(self):
        key = self._key_after(
            "    rng = random.Random(seed)",
            "    rng = random.Random(seed)  # source-edit-proof",
            "build_pool",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_family_fundamental_hz_moves_the_key(self):
        key = self._key_after(
            "    nyquist = sample_rate / 2.0",
            "    nyquist = sample_rate / 2.0  # source-edit-proof",
            "family_fundamental_hz",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_family_harmonic_gains_moves_the_key(self):
        key = self._key_after(
            "    raw = [1.0 / (1.0 + ((h + family) % _HARMONICS)) for h in range(_HARMONICS)]",
            "    raw = [1.0 / (1.0 + ((h + family) % _HARMONICS)) for h in range(_HARMONICS)]"
            "  # source-edit-proof",
            "family_harmonic_gains",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_instance_samples_moves_the_key(self):
        key = self._key_after(
            '    out = array.array("h", bytes(_SAMPLE_WIDTH_BYTES * frames))',
            '    out = array.array("h", bytes(_SAMPLE_WIDTH_BYTES * frames))  # source-edit-proof',
            "instance_samples",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_clip_name_moves_the_key(self):
        key = self._key_after(
            '    return f"clip_f{family:02d}_i{instance:03d}.wav"',
            '    return f"clip_f{family:02d}_i{instance:03d}.wav"  # source-edit-proof',
            "clip_name",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_encode_wav_moves_the_key(self):
        key = self._key_after(
            "    buf = io.BytesIO()",
            "    buf = io.BytesIO()  # source-edit-proof",
            "encode_wav",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_peak_moves_the_key(self):
        key = self._key_after("_PEAK = 12000", "_PEAK = 12001", "peak")
        self.assertNotEqual(self.baseline, key)

    def test_editing_phase_divisor_moves_the_key(self):
        key = self._key_after("_PHASE_DIVISOR = 256.0", "_PHASE_DIVISOR = 257.0", "phase_divisor")
        self.assertNotEqual(self.baseline, key)

    def test_editing_harmonics_moves_the_key(self):
        key = self._key_after("_HARMONICS = 4", "_HARMONICS = 5", "harmonics")
        self.assertNotEqual(self.baseline, key)

    def test_editing_sample_width_bytes_moves_the_key(self):
        key = self._key_after("_SAMPLE_WIDTH_BYTES = 2", "_SAMPLE_WIDTH_BYTES = 3", "sample_width")
        self.assertNotEqual(self.baseline, key)

    def test_editing_channels_moves_the_key(self):
        key = self._key_after("_CHANNELS = 1", "_CHANNELS = 2", "channels")
        self.assertNotEqual(self.baseline, key)

    def test_editing_read_wav_moves_the_key(self):
        """`read_wav` decodes bytes this producer already wrote and plays no
        part in constructing them -- yet the key is the WHOLE module's
        bytes, so editing it moves the key exactly as editing a function
        the pool construction actually calls does (see the design note on
        the class docstring: this fingerprint tracks no smaller a unit than
        the file)."""
        key = self._key_after(
            '    with wave.open(io.BytesIO(data), "rb") as w:',
            '    with wave.open(io.BytesIO(data), "rb") as w:  # source-edit-proof',
            "read_wav",
        )
        self.assertNotEqual(self.baseline, key)

    def test_editing_the_module_docstring_moves_the_key(self):
        key = self._key_after(
            "LENGTH GUARANTEE (issue #421 PR B's pre-registered training-step profile):",
            "LENGTH GUARANTEE EDITED (issue #421 PR B's pre-registered training-step profile):",
            "docstring",
        )
        self.assertNotEqual(self.baseline, key)


class FractionalSecondsTests(unittest.TestCase):
    """`--seconds` is a FLOAT and the #421 profile's declared audio shape is
    9.5 s at 48 kHz -- a fractional value on purpose (strictly below the CLAP
    front end's `nb_max_samples`, so the repeat-pad branch is the declared
    branch rather than a boundary case).

    The expected frame counts here are stated as LITERALS computed by hand
    (`9.5 * 48000 = 456_000`), never as `gfa.frame_count(...)` -- a test that
    asked the producer what it produces would agree with itself after any
    drift in the rounding rule.
    """

    def test_nine_point_five_seconds_at_48k_is_exactly_456000_frames(self):
        files, _rows = gfa.generate_corpus(
            rows=2, seconds=9.5, sample_rate=48000, seed=2, families=2, instances_per_family=2
        )
        self.assertTrue(files)
        for name, data in files.items():
            _channels, _width, rate, samples = gfa.read_wav(data)
            self.assertEqual(rate, 48000, name)
            self.assertEqual(
                len(samples), 456_000,
                f"{name}: 9.5 s at 48 kHz must be exactly 456000 frames, got {len(samples)}",
            )

    def test_other_fractional_durations_round_to_the_stated_frame_count(self):
        # Hand-computed: 0.125 * 16000 = 2000; 1.5 * 44100 = 66150;
        # 0.3 * 48000 = 14400 (0.3 is not exactly representable in binary
        # floating point, so this also pins that `round` -- not `int` --
        # is what the producer applies: `int(0.3 * 48000)` is 14399).
        for seconds, rate, expected in ((0.125, 16000, 2000), (1.5, 44100, 66150), (0.3, 48000, 14400)):
            files, _rows = gfa.generate_corpus(
                rows=2, seconds=seconds, sample_rate=rate, seed=1,
                families=2, instances_per_family=2,
            )
            for name, data in files.items():
                _c, _w, _r, samples = gfa.read_wav(data)
                self.assertEqual(
                    len(samples), expected,
                    f"{name}: {seconds} s at {rate} Hz must be {expected} frames",
                )


class HeldOutSplitTests(unittest.TestCase):
    """`--heldout-rows` (issue #421 P1-b(iv)) -- the audio twin of
    `test_gen_fixed_shape_image_corpus.py`'s own `HeldOutSplitTests`, making
    the SAME assertions against this producer's own emitted tree.

    Disjointness is asserted on the FILE SETS the two row lists actually
    reference, not on the family indices used internally, so it holds
    against what a consumer reads.
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

    def _split(self, **kw):
        args = dict(rows=12, seconds=0.05, sample_rate=16000, seed=3, families=6,
                    heldout_rows=4, heldout_batch=2)
        args.update(kw)
        return gfa.generate_split(**args)

    def test_row_count_is_exactly_what_was_asked_for(self):
        _files, train, heldout = self._split()
        self.assertEqual(len(train), 12)
        self.assertEqual(len(heldout), 4)

    def test_the_two_splits_reference_disjoint_files_and_disjoint_ids(self):
        _files, train, heldout = self._split()
        train_files = self._files_referenced(train)
        heldout_files = self._files_referenced(heldout)
        self.assertTrue(train_files)
        self.assertTrue(heldout_files)
        self.assertEqual(
            train_files & heldout_files,
            set(),
            "a family-disjoint split must share NO clip between the two halves",
        )
        self.assertEqual(self._ids(train) & self._ids(heldout), set())

    def test_every_referenced_file_actually_exists_in_the_emitted_corpus(self):
        files, train, heldout = self._split()
        for name in self._files_referenced(train) | self._files_referenced(heldout):
            self.assertIn(name, files, f"{name} is referenced but was never emitted")

    def test_heldout_rows_carry_the_train_schema_and_the_pinned_length(self):
        files, _train, heldout = self._split(seconds=0.1, sample_rate=16000)
        for row in heldout:
            self.assertEqual(set(row.keys()), _EXPECTED_KEYS)
            for key in ("anchor_path", "positive_path", "negative_path"):
                _c, _w, rate, samples = gfa.read_wav(files[row[key]])
                self.assertEqual(rate, 16000)
                self.assertEqual(len(samples), 1600)

    def test_a_heldout_row_is_a_real_triplet_same_family_positive(self):
        _files, _train, heldout = self._split()
        for row in heldout:
            anchor_fam = row["anchor_path"].split("_")[1]
            positive_fam = row["positive_path"].split("_")[1]
            negative_fam = row["negative_path"].split("_")[1]
            self.assertEqual(anchor_fam, positive_fam, row)
            self.assertNotEqual(anchor_fam, negative_fam, row)

    def test_the_audio_bytes_are_unchanged_by_a_heldout_request(self):
        files_plain, train_plain = gfa.generate_corpus(
            rows=12, seconds=0.05, sample_rate=16000, seed=3, families=6
        )
        files_split, train_split, _heldout = self._split()
        self.assertEqual(files_plain, files_split, "the emitted clip bytes must not change")
        self.assertNotEqual(
            train_plain,
            train_split,
            "the train rows MUST narrow to the unreserved families -- identical row lists would "
            "mean the reservation did nothing and the split is not actually disjoint",
        )

    def test_same_seed_same_bytes(self):
        self.assertEqual(self._split(seed=11), self._split(seed=11))

    def test_refuses_when_the_family_pool_cannot_support_the_split(self):
        with self.assertRaises(ValueError) as ctx:
            self._split(rows=4, families=3, heldout_rows=2)
        self.assertIn("--families", str(ctx.exception))
        # Boundary: 4 families reserving 2 leaves exactly 2 -- accepted, so
        # the refusal above is a real bound, not an off-by-one.
        _files, train, heldout = self._split(rows=4, families=4, heldout_rows=2)
        self.assertEqual((len(train), len(heldout)), (4, 2))

    def test_refuses_a_heldout_family_count_below_two(self):
        with self.assertRaises(ValueError) as ctx:
            self._split(heldout_families=1)
        self.assertIn("--heldout-families", str(ctx.exception))

    def test_refuses_a_heldout_row_count_that_is_not_a_multiple_of_the_batch(self):
        with self.assertRaises(ValueError) as ctx:
            self._split(heldout_rows=5, heldout_batch=2)
        self.assertIn("--heldout-batch", str(ctx.exception))

    def test_refuses_a_heldout_request_with_no_batch_stated(self):
        with self.assertRaises(ValueError) as ctx:
            self._split(heldout_batch=None)
        self.assertIn("--heldout-batch is required", str(ctx.exception))

    def test_cli_writes_both_heldout_files_and_they_agree_row_for_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "aud"
            rc = gfa.main([
                "--rows", "8",
                "--seconds", "0.05",
                "--sample-rate", "16000",
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
                self.assertEqual(
                    (anchor, positive, negative),
                    (row["anchor_id"], row["positive_id"], row["negative_id"]),
                )
            plain = Path(tmp) / "aud_plain"
            self.assertEqual(
                gfa.main([
                    "--rows", "8", "--seconds", "0.05", "--sample-rate", "16000",
                    "--seed", "5", "--out-dir", str(plain), "--families", "6",
                ]),
                0,
            )
            self.assertFalse((plain / "heldout_ids.txt").exists())
            self.assertFalse((plain / "heldout_triplets.jsonl").exists())

    def test_cli_bytes_are_deterministic_across_two_runs(self):
        def run(dirpath):
            self.assertEqual(
                gfa.main([
                    "--rows", "8", "--seconds", "0.05", "--sample-rate", "16000",
                    "--seed", "9", "--out-dir", str(dirpath), "--families", "6",
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
            rc = gfa.main([
                "--rows", "4", "--seconds", "0.05", "--sample-rate", "16000",
                "--seed", "1", "--out-dir", str(Path(tmp) / "x"), "--families", "3",
                "--heldout-rows", "2", "--heldout-batch", "2",
            ])
            self.assertEqual(rc, 2)
            self.assertFalse(
                (Path(tmp) / "x").exists(),
                "a refused split must write NOTHING -- not a partial corpus",
            )



if __name__ == "__main__":
    unittest.main()
