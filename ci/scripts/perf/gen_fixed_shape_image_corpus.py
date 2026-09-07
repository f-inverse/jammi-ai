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

Usage:
  gen_fixed_shape_image_corpus.py --rows N --size S --seed K --out-dir DIR
      [--families F] [--instances-per-family I] [--jitter J]
      [--jsonl-name NAME]

Hermetic: no network, writes only under `--out-dir`.
"""

from __future__ import annotations

import argparse
import binascii
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


def generate_corpus(
    rows: int,
    size: int,
    seed: int,
    families: int = _DEFAULT_FAMILIES,
    instances_per_family: int = _DEFAULT_INSTANCES_PER_FAMILY,
    jitter: int = _DEFAULT_JITTER,
) -> tuple[dict[str, bytes], list[dict]]:
    """Build the whole corpus in memory: `(files, rows)` where `files` maps a
    relative file name to its PNG bytes and `rows` is the JSONL row list.

    Pure with respect to its arguments -- no filesystem, no clock, no
    environment -- so determinism is testable without writing anything.
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

    rng = random.Random(seed)
    # Fixed draw order: family-major, instance-minor. Every later draw
    # depends only on the draws before it, so a larger `--instances-per-family`
    # run reproduces a smaller one's earlier images exactly.
    files: dict[str, bytes] = {}
    for family in range(families):
        template = _family_template(family, size)
        for instance in range(instances_per_family):
            pixels = _jittered(template, rng, jitter)
            files[_image_name(family, instance)] = encode_png(size, size, pixels)

    out_rows: list[dict] = []
    for i in range(rows):
        # Deterministic assignment, no RNG: row i walks the families and
        # instances in a fixed pattern, so the row list is a pure function of
        # `(rows, families, instances_per_family)` and never consumes from
        # `rng` (keeping the image bytes independent of `--rows`).
        fam = i % families
        neg_fam = (fam + 1 + (i // families) % (families - 1)) % families
        anchor_i = (2 * i) % instances_per_family
        positive_i = (anchor_i + 1) % instances_per_family
        negative_i = i % instances_per_family
        out_rows.append(
            {
                "anchor_id": f"img-{seed}-{i:06d}-a",
                "anchor_path": _image_name(fam, anchor_i),
                "positive_id": f"img-{seed}-{i:06d}-p",
                "positive_path": _image_name(fam, positive_i),
                "negative_id": f"img-{seed}-{i:06d}-n",
                "negative_path": _image_name(neg_fam, negative_i),
            }
        )
    return files, out_rows


def write_corpus(
    files: dict[str, bytes], rows: list[dict], out_dir: Path, jsonl_name: str
) -> Path:
    """Write the PNGs and the JSONL under `out_dir` (created if absent), in
    SORTED file-name order (family J: the emission order is fixed, never the
    dict's insertion order or the filesystem's). Returns the JSONL path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        (out_dir / name).write_bytes(files[name])
    jsonl_path = out_dir / jsonl_name
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
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
    args = ap.parse_args(argv)

    try:
        files, rows = generate_corpus(
            rows=args.rows,
            size=args.size,
            seed=args.seed,
            families=args.families,
            instances_per_family=args.instances_per_family,
            jitter=args.jitter,
        )
    except ValueError as e:
        print(f"::error::gen_fixed_shape_image_corpus: {e}", file=sys.stderr)
        return 2

    jsonl_path = write_corpus(files, rows, args.out_dir, args.jsonl_name)
    print(
        f"gen_fixed_shape_image_corpus: wrote {len(files)} PNGs of {args.size}x{args.size} RGB "
        f"and {len(rows)} triplet rows to {jsonl_path} "
        f"(families={args.families}, instances_per_family={args.instances_per_family}, "
        f"jitter={args.jitter}, seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
