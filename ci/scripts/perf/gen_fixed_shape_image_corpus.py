#!/usr/bin/env python3
"""Deterministic, seeded generator for a FIXED-SHAPE synthetic image triplet
corpus in the EXACT schema `jammi-bench finetune-run --train-jsonl
--task image_embedding` consumes (`crates/jammi-bench/src/main.rs::
MediaTripletRow`/`load_train_media_jsonl`, pinned by reading that source
directly, never guessed): one JSON object per line,

    {"anchor_id", "anchor_path", "positive_id", "positive_path",
     "negative_id", "negative_path"}

where every `*_path` is RELATIVE to the emitted JSONL's own directory (the
loader resolves it against that directory, so the corpus is relocatable as
one tree).

SHAPE GUARANTEE (issue #421 PR B's pre-registered training-step profile):
every emitted PNG is EXACTLY `--size x --size` RGB, 8 bits per channel, no
interlacing, no alpha. An OpenCLIP vision tower resizes whatever it is given
to `image_size` before the patch embedding; feeding it images that are
already exactly that shape means the measured wall carries the tower's real
fixed-shape front-end cost and NOTHING that varies row to row -- the
content-agnostic kernel wall the profile is pre-registered to compare. A
corpus of mixed-shape images would put a resample of varying cost inside
the timed region and make two legs' walls incomparable.

TRIPLET STRUCTURE: images are drawn from `--families` synthetic families.
A family is a deterministic *template* -- a periodic RGB pattern whose
spatial frequencies and per-channel phases are a pure function of the family
index -- and an INSTANCE of that family is the template plus seeded
per-pixel jitter of amplitude `--jitter` (default 8/255). Every row's anchor
and positive are two DISTINCT instances of the SAME family; the negative is
an instance of a DIFFERENT family. So "positive" and "negative" are
well-defined in pixel space by construction, and the test suite asserts the
separation MECHANICALLY (mean absolute intra-family distance strictly below
mean absolute inter-family distance) rather than assuming it. Nothing about
the content is claimed to be semantically meaningful: this is a fixed-shape
COST workload, not an accuracy fixture.

Determinism (family J): one `random.Random(seed)` instance draws every
jitter value in a single fixed sequential order -- families first, then
instances within a family, then rows -- so the same
`(size, families, instances, rows, jitter, seed)` tuple always produces
byte-identical PNGs AND a byte-identical JSONL. PNG bytes are additionally
pinned by writing at a FIXED zlib compression level (`_ZLIB_LEVEL`) with a
fixed filter byte (0, "None") on every scanline, so the encoder itself
contributes no run-to-run variation.

Generic fixture (family L): the content is synthetic periodic patterns plus
seeded noise. No consumer's data shape, no scraped imagery, no third-party
image library -- PNGs are written with a minimal encoder over the stdlib
`zlib`/`struct`/`binascii`, so this producer has NO dependency beyond the
Python standard library.

HELD-OUT SPLIT (issue #421 P1-b(iv)): `--heldout-rows N` additionally emits
`heldout_ids.txt` (TAB-separated `anchor_id\tpositive_id\tnegative_id`, one
row per line, in the order it was generated -- this file's ORDER is the
scoring identity `jammi-bench finetune-run --heldout-ids` reads) and
`heldout_triplets.jsonl` (the SAME row schema as the train JSONL), both in
`--out-dir` so the JSONL-relative `*_path` resolution the loader performs
(`crates/jammi-bench/src/main.rs::load_train_media_jsonl`, which resolves
each path against the JSONL's OWN directory) finds the same image files.

The split is disjoint BY FAMILY, never by seed: the LAST
`--heldout-families` of the `--families` pool are RESERVED for the held-out
rows and the train rows are drawn from the remaining ones, so no image file
referenced by a held-out row is ever referenced by a train row (a seed-based
"different draw" split would still share family templates, and two rows from
the same family are precisely what this producer calls a POSITIVE pair --
the held-out set would then be contaminated by construction). A `--families`
pool too small to give BOTH halves the two families a triplet needs is a
REFUSAL, never a silently overlapping split. `--heldout-batch B` is required
alongside `--heldout-rows` and refused unless it divides the held-out row
count exactly: `finetune-run` itself refuses a held-out fixture that is not
a nonzero multiple of `--batch`, and finding that out here (before any
image is written) is cheaper than finding it out on a GPU pod.

WITHOUT `--heldout-rows` (the default, 0) nothing about this producer's
output changes: the train rows are drawn from the FULL family pool exactly
as before, no extra files are written, and the emitted bytes are identical
to what every existing invocation already gets.

Usage:
  gen_fixed_shape_image_corpus.py --rows N --size S --seed K --out-dir DIR
      [--families F] [--instances-per-family I] [--jitter J]
      [--jsonl-name NAME]
      [--heldout-rows N --heldout-batch B [--heldout-families F]]
      [--pool-cache-dir DIR]

Hermetic: no network, writes only under `--out-dir` (and, only when
`--pool-cache-dir` is explicitly given, under that path too).

`--pool-cache-dir` (opt-in, default unset -- see that flag's own help):
lets many invocations sharing one `(--families, --instances-per-family,
--size, --jitter, --seed)` tuple synthesize the image pool ONCE instead of
once per invocation, with byte-identical output either way. A real leg
sweep never sets this; it exists for a test harness driving ~dozens of
hermetic invocations of this producer at the same pool shape.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import random
import struct
import sys
import zlib
from pathlib import Path

# PNG magic (8 bytes), fixed by the format.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Colour type 2 = truecolour RGB, 8 bits per channel.
_COLOR_TYPE_RGB = 2
_BIT_DEPTH = 8

# Fixed zlib level so the compressed IDAT bytes are a pure function of the
# raw scanlines -- determinism is a property of the emitted FILE, not just
# of the pixel array (family J). Level 6 is zlib's own default; naming it
# explicitly means a future change to that default cannot silently move
# every committed digest.
_ZLIB_LEVEL = 6

# Default per-pixel jitter amplitude, in 0-255 units. Small enough that an
# instance stays clearly inside its family's template (the intra < inter
# separation the test suite asserts), large enough that two instances of the
# same family are never byte-identical.
_DEFAULT_JITTER = 8

_DEFAULT_FAMILIES = 4
_DEFAULT_INSTANCES_PER_FAMILY = 4

# How many of `--families` are reserved for the held-out split by default.
# TWO is the floor, not a tuning knob: a triplet row needs a family for its
# anchor/positive and a DIFFERENT one for its negative, so a one-family
# held-out pool could not emit a well-formed row at all.
_DEFAULT_HELDOUT_FAMILIES = 2

# The names the held-out split is written under, inside `--out-dir`. Fixed
# (not flags): `heldout_ids.txt` is the id-ORDER file `finetune-run
# --heldout-ids` reads and `heldout_triplets.jsonl` is what
# `--heldout-jsonl` reads, and both must sit beside the images for the
# loader's JSONL-relative path resolution to find them.
_HELDOUT_IDS_NAME = "heldout_ids.txt"
_HELDOUT_JSONL_NAME = "heldout_triplets.jsonl"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: big-endian length, 4-byte tag, payload, CRC32 over
    tag+payload (the format's own definition -- the CRC does NOT cover the
    length field)."""
    crc = binascii.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png(width: int, height: int, pixels: bytes) -> bytes:
    """Encode `pixels` (`height * width * 3` bytes, row-major RGB) as an
    8-bit truecolour PNG.

    Every scanline is prefixed with filter byte 0 ("None"), so the raw
    stream is a pure, adaptive-heuristic-free function of the pixel bytes --
    a filter heuristic would make the output depend on the encoder version
    rather than on the image (family J).
    """
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"expected {expected} pixel bytes for {width}x{height} RGB, got {len(pixels)}")
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0: None
        raw += pixels[y * stride : (y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, _BIT_DEPTH, _COLOR_TYPE_RGB, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), _ZLIB_LEVEL))
        + _png_chunk(b"IEND", b"")
    )


def decode_png_rgb(data: bytes) -> tuple[int, int, bytes]:
    """Minimal inverse of [`encode_png`] for THIS producer's own output --
    8-bit RGB, filter 0 on every scanline, a single IDAT. Returns
    `(width, height, pixels)`.

    Exists so the test suite can assert the emitted files' SHAPE and CONTENT
    mechanically off the bytes actually written, rather than trusting the
    in-memory array the encoder was handed. Refuses (never guesses) on any
    PNG this producer would not itself have emitted.
    """
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG (bad signature)")
    pos = len(_PNG_SIGNATURE)
    width = height = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        (stored_crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        if stored_crc != (binascii.crc32(tag + payload) & 0xFFFFFFFF):
            raise ValueError(f"chunk {tag!r} has a bad CRC")
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (bit_depth, color_type, comp, filt, interlace) != (_BIT_DEPTH, _COLOR_TYPE_RGB, 0, 0, 0):
                raise ValueError("unsupported PNG variant for this decoder")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG has no IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    if len(raw) != height * (stride + 1):
        raise ValueError("PNG raw stream length does not match the declared geometry")
    out = bytearray()
    for y in range(height):
        f = raw[y * (stride + 1)]
        if f != 0:
            raise ValueError(f"scanline {y} uses filter {f}; this decoder only handles filter 0")
        out += raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)]
    return width, height, bytes(out)


def _family_template(family: int, size: int) -> bytes:
    """The deterministic, jitter-free RGB template for `family` at
    `size x size` -- a pure function of `(family, size)` with NO RNG, so two
    instances of the same family share a byte-identical base and the
    intra-vs-inter separation is a property of the construction rather than
    of a lucky draw."""
    # Spatial frequencies and per-channel phases derived from the family
    # index. Odd multipliers keep successive families from aliasing onto one
    # another at small `size`.
    fx = 1 + (family % 3)
    fy = 1 + ((family // 3) % 3)
    out = bytearray(size * size * 3)
    two_pi = 2.0 * math.pi
    for y in range(size):
        for x in range(size):
            base = (x * fx + y * fy) / max(size, 1)
            for c in range(3):
                phase = two_pi * ((family * 3 + c) % 7) / 7.0
                v = 128.0 + 100.0 * math.sin(two_pi * base + phase)
                out[(y * size + x) * 3 + c] = min(255, max(0, int(v)))
    return bytes(out)


def _jittered(template: bytes, rng: random.Random, jitter: int) -> bytes:
    """One INSTANCE: the template plus a per-byte integer draw in
    `[-jitter, jitter]`, clamped into `[0, 255]`. `jitter == 0` yields the
    template itself (and is refused at the CLI, since two instances of a
    family would then be byte-identical and no triplet row would carry a
    distinguishable positive)."""
    return bytes(
        min(255, max(0, b + rng.randint(-jitter, jitter))) for b in template
    )


def _image_name(family: int, instance: int) -> str:
    return f"img_f{family:02d}_i{instance:03d}.png"


def _build_rows(
    rows: int,
    seed: int,
    families: int,
    instances_per_family: int,
    family_offset: int,
    id_tag: str,
) -> list[dict]:
    """`rows` triplet rows over the family window
    `[family_offset, family_offset + families)`.

    Deterministic assignment, NO RNG: row `i` walks the families and
    instances in a fixed pattern, so the row list is a pure function of its
    arguments and never consumes from the image generator's `rng` (which is
    what keeps the image BYTES independent of `--rows`). `family_offset` is
    what makes the held-out split disjoint: the held-out call passes an
    offset past every family the train call can reach, so the two row lists
    name provably disjoint FILE sets. `id_tag` (`""` for train, `"h"` for
    held-out) keeps the two id spaces disjoint even at the same row index.
    """
    out_rows: list[dict] = []
    for i in range(rows):
        fam = family_offset + (i % families)
        neg_fam = family_offset + ((i % families) + 1 + (i // families) % (families - 1)) % families
        anchor_i = (2 * i) % instances_per_family
        positive_i = (anchor_i + 1) % instances_per_family
        negative_i = i % instances_per_family
        out_rows.append(
            {
                "anchor_id": f"img-{seed}-{id_tag}{i:06d}-a",
                "anchor_path": _image_name(fam, anchor_i),
                "positive_id": f"img-{seed}-{id_tag}{i:06d}-p",
                "positive_path": _image_name(fam, positive_i),
                "negative_id": f"img-{seed}-{id_tag}{i:06d}-n",
                "negative_path": _image_name(neg_fam, negative_i),
            }
        )
    return out_rows


def validate_heldout_split(
    families: int, heldout_families: int, heldout_rows: int, heldout_batch: int | None
) -> int:
    """Refuse a held-out request the family pool cannot support, and return
    the TRAIN family count (`families - heldout_families`).

    Every refusal here is a REFUSAL, never a silent degradation to an
    overlapping split: an overlapping "held-out" set shares family templates
    with the train set, and two same-family instances are exactly what this
    producer calls a POSITIVE pair -- so the leg would be scoring on rows it
    trained the family of.
    """
    if heldout_rows <= 0:
        raise ValueError(f"--heldout-rows must be positive when stated, got {heldout_rows}")
    if heldout_batch is None:
        raise ValueError(
            "--heldout-batch is required alongside --heldout-rows: `finetune-run` refuses a "
            "held-out fixture whose row count is not a nonzero multiple of --batch, and this "
            "producer cannot check that without being told the divisor"
        )
    if heldout_batch <= 0:
        raise ValueError(f"--heldout-batch must be positive, got {heldout_batch}")
    if heldout_rows % heldout_batch != 0:
        raise ValueError(
            f"--heldout-rows {heldout_rows} is not a multiple of --heldout-batch "
            f"{heldout_batch} (finetune-run refuses a held-out fixture that is not a nonzero "
            f"multiple of --batch)"
        )
    if heldout_families < 2:
        raise ValueError(
            f"--heldout-families must be at least 2 (a held-out triplet needs a DIFFERENT "
            f"family for its negative, exactly as a train triplet does), got {heldout_families}"
        )
    train_families = families - heldout_families
    if train_families < 2:
        raise ValueError(
            f"--families {families} cannot support a family-disjoint held-out split reserving "
            f"{heldout_families}: the train half would be left with {train_families} "
            f"famil{'y' if train_families == 1 else 'ies'}, and both halves need at least 2. "
            f"Raise --families to at least {heldout_families + 2}."
        )
    return train_families


def _build_pool(
    families: int, instances_per_family: int, size: int, jitter: int, seed: int
) -> dict[str, bytes]:
    """The family x instances image pool -- a pure function of exactly
    these five arguments (never `--rows`/`--heldout-*`/`--out-dir`/
    `--jsonl-name`), which is what makes it safe to cache: any invocation
    sharing this tuple, whatever it asks for downstream, gets the
    byte-identical pool this same code would have built inline. Factored
    out of `generate_split` so the cached and uncached paths run the
    EXACT SAME construction, never two implementations that could drift
    apart."""
    rng = random.Random(seed)
    files: dict[str, bytes] = {}
    for family in range(families):
        template = _family_template(family, size)
        for instance in range(instances_per_family):
            pixels = _jittered(template, rng, jitter)
            files[_image_name(family, instance)] = encode_png(size, size, pixels)
    return files


def _pool_cache_key(
    families: int, instances_per_family: int, size: int, jitter: int, seed: int
) -> str:
    """Filesystem-safe cache key: a sha256 of the producer MODULE'S OWN
    SOURCE BYTES (`Path(__file__).read_bytes()` -- the WHOLE file on disk)
    folded together with every argument `_build_pool` actually reads and
    the interpreter's `(major, minor)` version.

    Two module states on disk that differ by even one byte hash to two
    different keys, by construction: nothing about the file is walked,
    parsed, or classified, so there is no enumeration of "the functions
    that matter" to get wrong. The key is intentionally MAXIMAL -- an
    on-disk edit to ANY line of this module (a helper this call never
    reaches, a constant, a comment, even the module docstring) moves the
    key and costs one extra pool regeneration (seconds, in a test-only
    cache).

    Scope of the "no false cache hit" guarantee: it covers this file's
    bytes and the five arguments above, nothing else. A determinant of the
    bytes `_build_pool` produces that lives OUTSIDE this file and these
    arguments -- the zlib the interpreter links against, a different
    CPython implementation entirely -- is not folded into this key. That
    is safe only because `--pool-cache-dir` is per-process (the caller
    allocates it via `tempfile.mkdtemp` and tears it down at process exit
    -- see `test_profile_421_legs_dry_run.py`) and is refused outside
    `DRY_RUN` (`profile_421_legs.sh`'s own `POOL_CACHE_ARGS` guard): one
    invoking process, one interpreter, one platform, per cache directory,
    so there is never a second environment sharing that directory to
    collide against. `sys.version_info[:2]` is included anyway because
    the interpreter that runs `_build_pool` is as much a part of "what
    produced these bytes" as the source is, and it is the one such
    determinant this module can read directly."""
    module_bytes = Path(__file__).read_bytes()
    canonical = (
        f"module_sha256={hashlib.sha256(module_bytes).hexdigest()}|"
        f"py={sys.version_info[0]}.{sys.version_info[1]}|"
        f"families={families}|instances={instances_per_family}|size={size}|"
        f"jitter={jitter}|seed={seed}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


_POOL_CACHE_DONE_MARKER = "_DONE"


def _pool_marker_text(families: int, instances_per_family: int, size: int, jitter: int, seed: int) -> str:
    """The `_DONE` marker's own recorded-shape line -- the single
    definition [`_load_or_build_pool`] both WRITES on a cache miss and
    PARSES (via [`_parse_pool_marker`]) on a cache hit, so the two can
    never independently drift on the field set or the format."""
    return f"families={families} instances={instances_per_family} size={size} jitter={jitter} seed={seed}\n"


def _parse_pool_marker(text: str) -> dict[str, int]:
    """Inverse of [`_pool_marker_text`]: `"families=4 instances=3 ..."` ->
    `{"families": 4, "instances": 3, ...}`. Refuses (never guesses) on a
    marker that does not carry exactly the expected `key=int` tokens, so a
    hand-edited or truncated marker cannot be silently misread as a shape
    that happens to compare unequal-but-plausible."""
    fields: dict[str, int] = {}
    for token in text.split():
        key, sep, value = token.partition("=")
        if not sep or key in fields:
            raise ValueError(f"malformed or duplicate pool marker field {token!r} in {text!r}")
        fields[key] = int(value)
    return fields


def _load_or_build_pool(
    pool_cache_dir: Path | None,
    families: int,
    instances_per_family: int,
    size: int,
    jitter: int,
    seed: int,
) -> dict[str, bytes]:
    """[`_build_pool`] straight through when `pool_cache_dir` is `None`
    (real runs never set it -- see module doc's `--pool-cache-dir`) --
    IDENTICAL bytes to every invocation with `pool_cache_dir` set. With a
    cache dir given: a cache HIT reads every expected file straight off
    disk (never re-runs the pixel loop) -- but ONLY after re-checking the
    `_DONE` marker's own recorded shape against the shape THIS call was
    asked for; a cache MISS builds the pool once via `_build_pool`, writes
    it into a fresh per-key subdirectory, and only then drops the `_DONE`
    marker that makes it visible to a later hit -- so a process that dies
    mid-write leaves an incomplete (marker-less) subdirectory that the
    NEXT invocation rebuilds from scratch, rather than one that reads back
    a partial pool.

    The marker re-check is belt-and-braces at the point of use: `key`
    already identifies the requested shape uniquely (see
    `_pool_cache_key`'s own doc), so under that key's own guarantee the
    marker recorded under `cache_dir` can only ever describe THIS shape.
    Re-deriving the requested shape from the call's own arguments and
    comparing it, BY NAME, against what the marker actually recorded costs
    one dict comparison and catches, at the read site itself, any way a
    directory could come to hold a marker for a shape other than the one
    being asked for -- rather than trusting the key's uniqueness silently.
    """
    if pool_cache_dir is None:
        return _build_pool(families, instances_per_family, size, jitter, seed)

    key = _pool_cache_key(families, instances_per_family, size, jitter, seed)
    cache_dir = Path(pool_cache_dir) / key
    marker = cache_dir / _POOL_CACHE_DONE_MARKER
    expected_names = [
        _image_name(family, instance)
        for family in range(families)
        for instance in range(instances_per_family)
    ]
    if marker.is_file():
        recorded = _parse_pool_marker(marker.read_text())
        requested = _parse_pool_marker(
            _pool_marker_text(families, instances_per_family, size, jitter, seed)
        )
        if recorded != requested:
            raise ValueError(
                f"pool cache dir {cache_dir} is marked done for shape {recorded} but this call "
                f"requested shape {requested} under the SAME cache key {key!r} -- refusing to "
                "return a pool that may not match the requested shape rather than trusting the "
                "cache key's uniqueness silently"
            )
        return {name: (cache_dir / name).read_bytes() for name in expected_names}

    files = _build_pool(families, instances_per_family, size, jitter, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (cache_dir / name).write_bytes(data)
    marker.write_text(_pool_marker_text(families, instances_per_family, size, jitter, seed))
    return files


def generate_corpus(
    rows: int,
    size: int,
    seed: int,
    families: int = _DEFAULT_FAMILIES,
    instances_per_family: int = _DEFAULT_INSTANCES_PER_FAMILY,
    jitter: int = _DEFAULT_JITTER,
) -> tuple[dict[str, bytes], list[dict]]:
    """[`generate_split`] with NO held-out split — `(files, rows)`, the
    exact shape and values this function returned before `--heldout-rows`
    existed. Kept as the single-split entry point so every existing caller
    is untouched; `main` calls [`generate_split`] directly.
    """
    files, train_rows, _heldout = generate_split(
        rows=rows,
        size=size,
        seed=seed,
        families=families,
        instances_per_family=instances_per_family,
        jitter=jitter,
    )
    return files, train_rows


def generate_split(
    rows: int,
    size: int,
    seed: int,
    families: int = _DEFAULT_FAMILIES,
    instances_per_family: int = _DEFAULT_INSTANCES_PER_FAMILY,
    jitter: int = _DEFAULT_JITTER,
    heldout_rows: int = 0,
    heldout_families: int = _DEFAULT_HELDOUT_FAMILIES,
    heldout_batch: int | None = None,
    pool_cache_dir: Path | None = None,
) -> tuple[dict[str, bytes], list[dict], list[dict]]:
    """Build the whole corpus in memory: `(files, rows, heldout_rows_list)`
    where `files` maps a relative file name to its PNG bytes, `rows` is the
    train JSONL row list, and `heldout_rows_list` is the held-out one
    (EMPTY unless `heldout_rows > 0`).

    Pure with respect to its arguments -- no filesystem, no clock, no
    environment -- so determinism is testable without writing anything --
    UNLESS `pool_cache_dir` is given (opt-in only; every real invocation
    leaves it `None`), in which case the image POOL (never the row lists)
    is read from or written to that directory (see `_load_or_build_pool`)
    but the RETURNED bytes are, by construction, identical either way.
    """
    if rows <= 0:
        raise ValueError(f"--rows must be positive, got {rows}")
    if size <= 0:
        raise ValueError(f"--size must be positive, got {size}")
    if families < 2:
        raise ValueError(
            f"--families must be at least 2 (a triplet needs a DIFFERENT family for its "
            f"negative), got {families}"
        )
    if instances_per_family < 2:
        raise ValueError(
            f"--instances-per-family must be at least 2 (a row's anchor and positive are two "
            f"DISTINCT instances of one family), got {instances_per_family}"
        )
    if jitter < 1:
        raise ValueError(
            f"--jitter must be at least 1 (at 0 every instance of a family is byte-identical, "
            f"so a row's positive would be indistinguishable from its anchor), got {jitter}"
        )

    # The family pool is SPLIT only when a held-out set is requested; with
    # `--heldout-rows 0` (the default) `train_families == families` and
    # every line below runs exactly as it did before this flag existed.
    train_families = families
    if heldout_rows > 0:
        train_families = validate_heldout_split(
            families, heldout_families, heldout_rows, heldout_batch
        )

    # Fixed draw order: family-major, instance-minor. Every later draw
    # depends only on the draws before it, so a larger `--instances-per-family`
    # run reproduces a smaller one's earlier images exactly. `_build_pool`
    # is the SAME function whether or not `pool_cache_dir` is given (see
    # `_load_or_build_pool`'s own doc) -- caching can only skip re-running
    # this construction, never change what it would have produced.
    files = _load_or_build_pool(pool_cache_dir, families, instances_per_family, size, jitter, seed)

    out_rows = _build_rows(rows, seed, train_families, instances_per_family, 0, "")
    heldout_out_rows: list[dict] = []
    if heldout_rows > 0:
        # The held-out window starts where the train window ends, so the two
        # row lists cannot name a shared file.
        heldout_out_rows = _build_rows(
            heldout_rows, seed, heldout_families, instances_per_family, train_families, "h"
        )
    return files, out_rows, heldout_out_rows


def write_heldout(heldout_rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    """Write `heldout_ids.txt` + `heldout_triplets.jsonl` under `out_dir`,
    in the SAME row order, and return `(ids_path, jsonl_path)`.

    The ids file is the SCORING ORDER (`finetune-run --heldout-ids`), the
    JSONL is joined to it BY `anchor_id`; writing both from one list in one
    pass is what makes them consistent by construction rather than by
    convention."""
    ids_path = out_dir / _HELDOUT_IDS_NAME
    with ids_path.open("w") as f:
        for row in heldout_rows:
            f.write(f"{row['anchor_id']}\t{row['positive_id']}\t{row['negative_id']}\n")
    jsonl_path = out_dir / _HELDOUT_JSONL_NAME
    with jsonl_path.open("w") as f:
        for row in heldout_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return ids_path, jsonl_path


def write_corpus(
    files: dict[str, bytes],
    rows: list[dict],
    out_dir: Path,
    jsonl_name: str,
    heldout_rows: list[dict] | None = None,
) -> Path:
    """Write the PNGs and the JSONL under `out_dir` (created if absent), in
    SORTED file-name order (family J: the emission order is fixed, never the
    dict's insertion order or the filesystem's), plus the held-out pair of
    files when `heldout_rows` is non-empty. Returns the train JSONL path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        (out_dir / name).write_bytes(files[name])
    jsonl_path = out_dir / jsonl_name
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if heldout_rows:
        write_heldout(heldout_rows, out_dir)
    return jsonl_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="gen_fixed_shape_image_corpus.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rows", type=int, required=True, help="number of triplet rows")
    ap.add_argument(
        "--size", type=int, required=True, help="every emitted PNG is exactly SIZE x SIZE RGB"
    )
    ap.add_argument("--seed", type=int, required=True, help="deterministic RNG seed")
    ap.add_argument("--out-dir", type=Path, required=True, help="directory for the PNGs + JSONL")
    ap.add_argument("--families", type=int, default=_DEFAULT_FAMILIES)
    ap.add_argument("--instances-per-family", type=int, default=_DEFAULT_INSTANCES_PER_FAMILY)
    ap.add_argument(
        "--jitter",
        type=int,
        default=_DEFAULT_JITTER,
        help="per-pixel jitter amplitude in 0-255 units (see module doc)",
    )
    ap.add_argument("--jsonl-name", default="triplets.jsonl")
    ap.add_argument(
        "--heldout-rows",
        type=int,
        default=0,
        help="also emit a FAMILY-DISJOINT held-out split of this many rows "
        "(heldout_ids.txt + heldout_triplets.jsonl in --out-dir); 0 (the default) emits "
        "nothing extra and leaves every byte of the train corpus unchanged",
    )
    ap.add_argument(
        "--heldout-families",
        type=int,
        default=_DEFAULT_HELDOUT_FAMILIES,
        help="how many of --families to RESERVE for the held-out split (read only when "
        "--heldout-rows > 0); the train rows use the rest",
    )
    ap.add_argument(
        "--heldout-batch",
        type=int,
        default=None,
        help="required alongside --heldout-rows: the --batch the consuming finetune-run leg "
        "will use, which the held-out row count must be a nonzero multiple of",
    )
    ap.add_argument(
        "--pool-cache-dir",
        type=Path,
        default=None,
        help="OPT-IN (default: unset). When given, the family x instances image POOL -- a "
        "pure function of (--families, --instances-per-family, --size, --jitter, --seed) that "
        "never depends on --rows/--heldout-*/--out-dir -- is read from (or, on a first call, "
        "written to) a per-key subdirectory of this path instead of being re-encoded on every "
        "invocation. The cache key covers this WHOLE producer file's own source bytes (a "
        "sha256 of Path(__file__).read_bytes(), plus the interpreter's (major, minor) version "
        "and the pool-shape arguments above), so any on-disk edit to this module -- not only "
        "one that touches _build_pool's own call graph -- forces a fresh pool build. Emitted "
        "bytes are byte-identical with or without this flag (see "
        "test_gen_fixed_shape_image_corpus.py); real runs never set it -- it exists ONLY to let "
        "a test harness synthesize a shared pool once across many invocations of this producer "
        "at the same (families, instances, size, jitter, seed) tuple.",
    )
    args = ap.parse_args(argv)

    try:
        files, rows, heldout_rows = generate_split(
            rows=args.rows,
            size=args.size,
            seed=args.seed,
            families=args.families,
            instances_per_family=args.instances_per_family,
            jitter=args.jitter,
            heldout_rows=args.heldout_rows,
            heldout_families=args.heldout_families,
            heldout_batch=args.heldout_batch,
            pool_cache_dir=args.pool_cache_dir,
        )
    except ValueError as e:
        print(f"::error::gen_fixed_shape_image_corpus: {e}", file=sys.stderr)
        return 2

    jsonl_path = write_corpus(files, rows, args.out_dir, args.jsonl_name, heldout_rows)
    print(
        f"gen_fixed_shape_image_corpus: wrote {len(files)} PNGs of {args.size}x{args.size} RGB "
        f"and {len(rows)} triplet rows to {jsonl_path} "
        f"(families={args.families}, instances_per_family={args.instances_per_family}, "
        f"jitter={args.jitter}, seed={args.seed})"
    )
    if heldout_rows:
        print(
            f"gen_fixed_shape_image_corpus: wrote {len(heldout_rows)} held-out rows to "
            f"{args.out_dir / _HELDOUT_IDS_NAME} + {args.out_dir / _HELDOUT_JSONL_NAME} "
            f"(heldout_families={args.heldout_families} reserved from the family pool, "
            f"heldout_batch={args.heldout_batch})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
