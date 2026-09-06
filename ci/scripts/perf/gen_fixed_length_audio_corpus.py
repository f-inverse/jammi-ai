#!/usr/bin/env python3
"""Deterministic, seeded generator for a FIXED-LENGTH synthetic audio triplet
corpus in the EXACT schema `jammi-bench finetune-run --train-jsonl
--task audio_embedding` consumes (`crates/jammi-bench/src/main.rs::
MediaTripletRow`/`load_train_media_jsonl`, pinned by reading that source
directly, never guessed): one JSON object per line,

    {"anchor_id", "anchor_path", "positive_id", "positive_path",
     "negative_id", "negative_path"}

where every `*_path` is RELATIVE to the emitted JSONL's own directory (the
loader resolves it against that directory, so the corpus is relocatable as
one tree).

LENGTH GUARANTEE (issue #421 PR B's pre-registered training-step profile):
every emitted clip is EXACTLY `round(--seconds * --sample-rate)` frames of
16-bit signed little-endian PCM, single channel, at `--sample-rate` Hz --
asserted by this producer against the header the stdlib `wave` module
actually wrote, never merely intended. A CLAP audio front end folds a clip
into a FIXED fusion window before the HTSAT tower sees it, so a corpus whose
clips are all exactly one window long puts the tower's real fixed-shape
front-end cost -- and nothing that varies row to row -- inside the timed
region. Mixed-length clips would move a variable-cost pad/truncate/fuse step
into the wall and make two legs incomparable.

TRIPLET STRUCTURE: clips are drawn from `--families` synthetic families. A
family is a deterministic *timbre* -- a fundamental frequency and a harmonic
amplitude profile that are pure functions of the family index -- and an
INSTANCE of that family is that timbre at a per-instance phase offset plus
seeded sample noise of amplitude `--jitter` (in int16 units). Every row's
anchor and positive are two DISTINCT instances of the SAME family; the
negative is an instance of a DIFFERENT family. "Positive" and "negative" are
therefore well-defined in waveform space by construction, and the test suite
asserts the separation MECHANICALLY (mean absolute intra-family sample
distance strictly below mean absolute inter-family distance) rather than
assuming it. Nothing here is claimed to be semantically meaningful: this is
a fixed-length COST workload, not an accuracy fixture.

Determinism (family J): one `random.Random(seed)` instance draws every noise
sample in a single fixed sequential order -- families first, then instances
within a family -- so the same `(seconds, sample_rate, families, instances,
jitter, seed)` tuple always produces byte-identical WAVs AND a
byte-identical JSONL. The row list consumes no RNG at all, so `--rows` never
perturbs the audio bytes.

Generic fixture (family L): the content is synthetic additive-harmonic tones
plus seeded noise. No consumer's data, no recorded audio, no third-party
package -- clips are written with the stdlib `wave` module over `array`, so
this producer has NO dependency beyond the Python standard library.

Usage:
  gen_fixed_length_audio_corpus.py --rows N --seconds T --sample-rate R
      --seed K --out-dir DIR [--families F] [--instances-per-family I]
      [--jitter J] [--jsonl-name NAME]

Hermetic: no network, writes only under `--out-dir`.
"""

from __future__ import annotations

import argparse
import array
import io
import json
import math
import random
import sys
import wave
from pathlib import Path

# 16-bit signed PCM, mono -- the shape every `*_path` in the emitted JSONL
# carries. Named constants so the assertions below and the module doc's
# guarantee cannot drift apart.
_SAMPLE_WIDTH_BYTES = 2
_CHANNELS = 1

# int16 headroom left for the additive harmonic stack before jitter is
# added, so a clip never clips (a saturated waveform would make two families'
# instances converge on the same rail and quietly weaken the intra-vs-inter
# separation the triplet structure depends on).
_PEAK = 12000

_DEFAULT_JITTER = 200
_DEFAULT_FAMILIES = 4
_DEFAULT_INSTANCES_PER_FAMILY = 4

# Harmonics summed per family. Fixed, not a knob: the profile's workload is
# pinned by the committed producer invocation, and a per-run harmonic count
# would be one more axis a reader must reconcile between two legs.
_HARMONICS = 4

# Instance `i` of a family is that family's timbre rotated by
# `2*pi*i/_PHASE_DIVISOR` radians. Large enough that two instances are
# genuinely distinct waveforms, small enough that an intra-family pair stays
# nearer to each other than to any other family's clip.
_PHASE_DIVISOR = 256.0


def _family_fundamental_hz(family: int, sample_rate: int) -> float:
    """A family's fundamental, as a pure function of `(family, sample_rate)`.

    Anchored to the sample rate rather than to an absolute Hz value so the
    generated waveform occupies the same fraction of the Nyquist band at any
    `--sample-rate`, and so the top harmonic (`_HARMONICS * f0`) stays below
    Nyquist for every family this producer will emit.
    """
    # 1/128 .. of Nyquist, spread across families; `_HARMONICS * f0` is then
    # at most `_HARMONICS * (families + 1) / 128` of Nyquist, which stays
    # under 1.0 for every family count a caller can reasonably pass.
    nyquist = sample_rate / 2.0
    return nyquist * (family + 1) / 128.0


def _family_harmonic_gains(family: int) -> list[float]:
    """Per-harmonic amplitude weights for a family, normalised to sum to 1 so
    every family's clean waveform has the same peak budget (`_PEAK`) and the
    intra-vs-inter separation reflects TIMBRE, not loudness."""
    raw = [1.0 / (1.0 + ((h + family) % _HARMONICS)) for h in range(_HARMONICS)]
    total = sum(raw)
    return [g / total for g in raw]


def _instance_samples(
    family: int, instance: int, frames: int, sample_rate: int, rng: random.Random, jitter: int
) -> array.array:
    """One INSTANCE's int16 sample array: the family timbre at a
    per-instance phase offset, plus a per-sample integer draw in
    `[-jitter, jitter]`, clamped into the int16 range."""
    f0 = _family_fundamental_hz(family, sample_rate)
    gains = _family_harmonic_gains(family)
    # A SMALL per-instance phase offset. Two instances of one family must
    # stay clearly closer to each other than to any other family's instance
    # (the intra < inter separation the test suite asserts mechanically), so
    # the offset is a fraction of a cycle, not a large rotation that would
    # make an intra-family pair as far apart as an inter-family one.
    phase = 2.0 * math.pi * instance / _PHASE_DIVISOR
    out = array.array("h", bytes(_SAMPLE_WIDTH_BYTES * frames))
    two_pi_over_sr = 2.0 * math.pi / sample_rate
    for n in range(frames):
        acc = 0.0
        for h, gain in enumerate(gains):
            acc += gain * math.sin(two_pi_over_sr * f0 * (h + 1) * n + phase)
        value = int(_PEAK * acc) + rng.randint(-jitter, jitter)
        out[n] = min(32767, max(-32768, value))
    return out


def encode_wav(samples: array.array, sample_rate: int) -> bytes:
    """Encode int16 mono `samples` as a RIFF/WAVE file via the stdlib `wave`
    module. The bytes are a pure function of the samples and the rate --
    `wave` writes no timestamp, no encoder string, nothing environmental
    (family J)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(_CHANNELS)
        w.setsampwidth(_SAMPLE_WIDTH_BYTES)
        w.setframerate(sample_rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[int, int, int, array.array]:
    """Inverse of [`encode_wav`] for this producer's own output: returns
    `(channels, sample_width, sample_rate, samples)` read back off the
    BYTES, so the test suite can assert the emitted length/rate against the
    header actually written rather than the value that was intended."""
    with wave.open(io.BytesIO(data), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    if width != _SAMPLE_WIDTH_BYTES:
        raise ValueError(f"expected {_SAMPLE_WIDTH_BYTES}-byte samples, got {width}")
    samples = array.array("h")
    samples.frombytes(frames)
    return channels, width, rate, samples


def frame_count(seconds: float, sample_rate: int) -> int:
    """The EXACT frame count every clip carries: `round(seconds *
    sample_rate)`. One definition, used by the generator and re-asserted by
    the test suite against the written header -- never two independent
    roundings that could disagree."""
    return int(round(seconds * sample_rate))


def _clip_name(family: int, instance: int) -> str:
    return f"clip_f{family:02d}_i{instance:03d}.wav"


def generate_corpus(
    rows: int,
    seconds: float,
    sample_rate: int,
    seed: int,
    families: int = _DEFAULT_FAMILIES,
    instances_per_family: int = _DEFAULT_INSTANCES_PER_FAMILY,
    jitter: int = _DEFAULT_JITTER,
) -> tuple[dict[str, bytes], list[dict]]:
    """Build the whole corpus in memory: `(files, rows)` where `files` maps a
    relative file name to its WAV bytes and `rows` is the JSONL row list.

    Pure with respect to its arguments -- no filesystem, no clock, no
    environment -- so determinism is testable without writing anything.
    """
    if rows <= 0:
        raise ValueError(f"--rows must be positive, got {rows}")
    if sample_rate <= 0:
        raise ValueError(f"--sample-rate must be positive, got {sample_rate}")
    if seconds <= 0:
        raise ValueError(f"--seconds must be positive, got {seconds}")
    frames = frame_count(seconds, sample_rate)
    if frames <= 0:
        raise ValueError(
            f"--seconds {seconds} at --sample-rate {sample_rate} rounds to {frames} frames; "
            f"a clip must carry at least one frame"
        )
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
            f"--jitter must be at least 1 (at 0 two instances of a family differ only by a "
            f"phase offset and a row's positive stops being a distinct recording), got {jitter}"
        )

    rng = random.Random(seed)
    files: dict[str, bytes] = {}
    # Fixed draw order: family-major, instance-minor (family J).
    for family in range(families):
        for instance in range(instances_per_family):
            samples = _instance_samples(family, instance, frames, sample_rate, rng, jitter)
            files[_clip_name(family, instance)] = encode_wav(samples, sample_rate)

    out_rows: list[dict] = []
    for i in range(rows):
        # Deterministic assignment, no RNG (so `--rows` never perturbs the
        # audio bytes) -- the same walk `gen_fixed_shape_image_corpus.py`
        # uses, so the two media corpora pair row-for-row by index.
        fam = i % families
        neg_fam = (fam + 1 + (i // families) % (families - 1)) % families
        anchor_i = (2 * i) % instances_per_family
        positive_i = (anchor_i + 1) % instances_per_family
        negative_i = i % instances_per_family
        out_rows.append(
            {
                "anchor_id": f"aud-{seed}-{i:06d}-a",
                "anchor_path": _clip_name(fam, anchor_i),
                "positive_id": f"aud-{seed}-{i:06d}-p",
                "positive_path": _clip_name(fam, positive_i),
                "negative_id": f"aud-{seed}-{i:06d}-n",
                "negative_path": _clip_name(neg_fam, negative_i),
            }
        )
    return files, out_rows


def write_corpus(
    files: dict[str, bytes], rows: list[dict], out_dir: Path, jsonl_name: str
) -> Path:
    """Write the WAVs and the JSONL under `out_dir` (created if absent), in
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
        prog="gen_fixed_length_audio_corpus.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rows", type=int, required=True, help="number of triplet rows")
    ap.add_argument(
        "--seconds", type=float, required=True, help="every clip is exactly this long"
    )
    ap.add_argument("--sample-rate", type=int, required=True, help="frames per second")
    ap.add_argument("--seed", type=int, required=True, help="deterministic RNG seed")
    ap.add_argument("--out-dir", type=Path, required=True, help="directory for the WAVs + JSONL")
    ap.add_argument("--families", type=int, default=_DEFAULT_FAMILIES)
    ap.add_argument("--instances-per-family", type=int, default=_DEFAULT_INSTANCES_PER_FAMILY)
    ap.add_argument(
        "--jitter",
        type=int,
        default=_DEFAULT_JITTER,
        help="per-sample noise amplitude in int16 units (see module doc)",
    )
    ap.add_argument("--jsonl-name", default="triplets.jsonl")
    args = ap.parse_args(argv)

    try:
        files, rows = generate_corpus(
            rows=args.rows,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
            seed=args.seed,
            families=args.families,
            instances_per_family=args.instances_per_family,
            jitter=args.jitter,
        )
    except ValueError as e:
        print(f"::error::gen_fixed_length_audio_corpus: {e}", file=sys.stderr)
        return 2

    jsonl_path = write_corpus(files, rows, args.out_dir, args.jsonl_name)
    frames = frame_count(args.seconds, args.sample_rate)
    print(
        f"gen_fixed_length_audio_corpus: wrote {len(files)} clips of {frames} frames "
        f"({args.seconds}s at {args.sample_rate} Hz, 16-bit mono PCM) and {len(rows)} triplet "
        f"rows to {jsonl_path} (families={args.families}, "
        f"instances_per_family={args.instances_per_family}, jitter={args.jitter}, "
        f"seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
