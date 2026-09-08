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

FRACTIONAL `--seconds`: `--seconds` is a float and the clip length is
`round(seconds * sample_rate)` frames ([`frame_count`], one definition used
by the generator AND re-asserted by the test suite off the written WAV
header). `--seconds 9.5 --sample-rate 48000` is therefore exactly 456000
frames -- the #421 profile's declared audio shape, chosen to sit strictly
BELOW the CLAP front end's `nb_max_samples` so the repeat-pad branch is the
declared branch rather than a boundary case.

HELD-OUT SPLIT (issue #421 P1-b(iv)): `--heldout-rows N` additionally emits
`heldout_ids.txt` (TAB-separated `anchor_id\tpositive_id\tnegative_id`, one
row per line, in the order it was generated -- this file's ORDER is the
scoring identity `jammi-bench finetune-run --heldout-ids` reads) and
`heldout_triplets.jsonl` (the SAME row schema as the train JSONL), both in
`--out-dir` so the JSONL-relative `*_path` resolution the loader performs
(`crates/jammi-bench/src/main.rs::load_train_media_jsonl`, which resolves
each path against the JSONL's OWN directory) finds the same clip files.

The split is disjoint BY FAMILY, never by seed: the LAST
`--heldout-families` of the `--families` pool are RESERVED for the held-out
rows and the train rows are drawn from the remaining ones, so no clip
referenced by a held-out row is ever referenced by a train row (a seed-based
"different draw" split would still share family timbres, and two clips of
one timbre are precisely what this producer calls a POSITIVE pair -- the
held-out set would be contaminated by construction). A `--families` pool too
small to give BOTH halves the two families a triplet needs is a REFUSAL,
never a silently overlapping split. `--heldout-batch B` is required
alongside `--heldout-rows` and refused unless it divides the held-out row
count exactly: `finetune-run` itself refuses a held-out fixture that is not
a nonzero multiple of `--batch`, and finding that out here (before any WAV
is written) is cheaper than finding it out on a GPU pod.

WITHOUT `--heldout-rows` (the default, 0) nothing about this producer's
output changes: the train rows are drawn from the FULL family pool exactly
as before, no extra files are written, and the emitted bytes are identical
to what every existing invocation already gets.

Usage:
  gen_fixed_length_audio_corpus.py --rows N --seconds T --sample-rate R
      --seed K --out-dir DIR [--families F] [--instances-per-family I]
      [--jitter J] [--jsonl-name NAME]
      [--heldout-rows N --heldout-batch B [--heldout-families F]]
      [--pool-cache-dir DIR]

Hermetic: no network, writes only under `--out-dir` (and, only when
`--pool-cache-dir` is explicitly given, under that path too).

`--pool-cache-dir` (opt-in, default unset -- see that flag's own help):
lets many invocations sharing one `(--families, --instances-per-family,
--seconds, --sample-rate, --jitter, --seed)` tuple synthesize the audio
pool ONCE instead of once per invocation, with byte-identical output
either way. A real leg sweep never sets this; it exists for a test harness
driving ~dozens of hermetic invocations of this producer at the same pool
shape.
"""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import inspect
import io
import json
import math
import random
import sys
import textwrap
import types
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


# How many of `--families` are reserved for the held-out split by default.
# TWO is the floor, not a tuning knob: a triplet row needs a family for its
# anchor/positive and a DIFFERENT one for its negative, so a one-family
# held-out pool could not emit a well-formed row at all.
_DEFAULT_HELDOUT_FAMILIES = 2

# The names the held-out split is written under, inside `--out-dir` -- the
# SAME two names `gen_fixed_shape_image_corpus.py` emits, so a driver
# handles both modalities with one pair of paths.
_HELDOUT_IDS_NAME = "heldout_ids.txt"
_HELDOUT_JSONL_NAME = "heldout_triplets.jsonl"


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

    Deterministic assignment, NO RNG (so `--rows` never perturbs the audio
    bytes) -- the same walk `gen_fixed_shape_image_corpus.py` uses, so the
    two media corpora pair row-for-row by index. `family_offset` is what
    makes the held-out split disjoint: the held-out call passes an offset
    past every family the train call can reach, so the two row lists name
    provably disjoint FILE sets. `id_tag` (`""` for train, `"h"` for
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
                "anchor_id": f"aud-{seed}-{id_tag}{i:06d}-a",
                "anchor_path": _clip_name(fam, anchor_i),
                "positive_id": f"aud-{seed}-{id_tag}{i:06d}-p",
                "positive_path": _clip_name(fam, positive_i),
                "negative_id": f"aud-{seed}-{id_tag}{i:06d}-n",
                "negative_path": _clip_name(neg_fam, negative_i),
            }
        )
    return out_rows


def validate_heldout_split(
    families: int, heldout_families: int, heldout_rows: int, heldout_batch: int | None
) -> int:
    """Refuse a held-out request the family pool cannot support, and return
    the TRAIN family count (`families - heldout_families`).

    Every refusal here is a REFUSAL, never a silent degradation to an
    overlapping split: an overlapping "held-out" set shares family timbres
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
    families: int,
    instances_per_family: int,
    frames: int,
    sample_rate: int,
    jitter: int,
    seed: int,
) -> dict[str, bytes]:
    """The family x instances audio pool -- a pure function of exactly
    these six arguments (never `--rows`/`--heldout-*`/`--out-dir`/
    `--jsonl-name`), which is what makes it safe to cache: any invocation
    sharing this tuple, whatever it asks for downstream, gets the
    byte-identical pool this same code would have built inline. Factored
    out of `generate_split` so the cached and uncached paths run the EXACT
    SAME construction, never two implementations that could drift apart."""
    rng = random.Random(seed)
    files: dict[str, bytes] = {}
    for family in range(families):
        for instance in range(instances_per_family):
            samples = _instance_samples(family, instance, frames, sample_rate, rng, jitter)
            files[_clip_name(family, instance)] = encode_wav(samples, sample_rate)
    return files


def _referenced_global_names(fn) -> set[str]:
    """The module-level names `fn`'s own SOURCE TEXT loads, parsed fresh via
    `ast` on every call -- never a hand-maintained list that can go stale the
    moment `fn`'s body changes. Excludes every name `fn` itself BINDS
    (parameters, locals, loop/comprehension targets), so a name that merely
    shadows a module global inside the function is not mistaken for a
    reference to it."""
    fn_def = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    bound = {n.arg for n in ast.walk(fn_def) if isinstance(n, ast.arg)}
    bound |= {
        n.id for n in ast.walk(fn_def) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    loaded = {
        n.id for n in ast.walk(fn_def) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return loaded - bound


def _pool_construction_closure() -> tuple[list[str], list[str]]:
    """The TRANSITIVE CLOSURE of module-level functions and constants
    `_build_pool` reaches, derived by walking its own source (and the source
    of everything it reaches, recursively) rather than read off a
    hand-maintained tuple -- the same graph an auditor would draw by hand,
    rebuilt fresh by the code itself on every call so a name can never fall
    out of it by someone forgetting to add it.

    A referenced name that resolves (via `globals()`) to a module-level
    FUNCTION is recursed into; one that resolves to an imported MODULE or a
    class is skipped (neither is "this module's construction code", and a
    module's own `repr()` embeds its filesystem path, which would make the
    fingerprint move across machines for no code reason); anything else --
    an int, a str, a tuple, a bytes literal -- is a CONSTANT, hashed by
    value. Returns `(function_names, constant_names)`, both sorted.
    """
    g = globals()
    seen_functions: set[str] = set()
    constant_names: set[str] = set()
    worklist = ["_build_pool"]
    while worklist:
        name = worklist.pop()
        if name in seen_functions:
            continue
        seen_functions.add(name)
        for ref in _referenced_global_names(g[name]):
            if ref not in g:
                continue  # builtin, or a name this function's own scope binds
            value = g[ref]
            if inspect.isfunction(value) and getattr(value, "__globals__", None) is g:
                if ref not in seen_functions:
                    worklist.append(ref)
            elif isinstance(value, types.ModuleType) or inspect.isclass(value):
                continue
            else:
                constant_names.add(ref)
    return sorted(seen_functions), sorted(constant_names)


def _pool_construction_fingerprint() -> str:
    """Hex digest of the ACTUAL construction code, not a hand-maintained
    version string: source text (`inspect.getsource`) of every function in
    [`_pool_construction_closure`]'s function set, plus the `repr()` of every
    constant in its constant set, looked up fresh via `globals()` on every
    call.

    Hashing the whole MODULE FILE's bytes would also be a complete fix (it
    covers everything below, plus every byte outside the construction path
    -- CLI parsing, docstrings, the held-out-split logic -- that cannot
    possibly change emitted waveform/WAV bytes), but a test cannot observe
    it responding to a live code change: `Path(__file__).read_bytes()`
    reads the file on DISK, which a `unittest.mock`/monkeypatch of a
    module-level function or constant never touches, so a regression that
    quietly changes what `_build_pool` emits without a matching source edit
    would pass such a test the same way it passed the old hand-bumped
    `_PRODUCER_VERSION` scheme this replaces. Hashing live function/constant
    OBJECTS via `globals()` fixes that: monkeypatching `_instance_samples`
    (or any name the closure reaches, `_build_pool` and `_clip_name`
    included) changes what THIS function reads on its very next call, so a
    test can assert the cache key moves under a patch and stays put when
    nothing relevant changed -- the property this replacement exists to
    prove."""
    g = globals()
    function_names, constant_names = _pool_construction_closure()
    parts = [inspect.getsource(g[name]) for name in function_names]
    parts.append(repr(tuple(g[name] for name in constant_names)))
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()[:32]


def _pool_cache_key(
    families: int,
    instances_per_family: int,
    frames: int,
    sample_rate: int,
    jitter: int,
    seed: int,
) -> str:
    """Filesystem-safe cache key over every argument `_build_pool` actually
    reads (`frames`, not the raw `--seconds` float, since `frames` is what
    the pool construction itself consumes -- see `frame_count`), plus
    `_pool_construction_fingerprint()` (the construction CODE itself, not a
    hand-bumped version string) -- omitting any one of these would let two
    genuinely different pools collide on the same cache directory."""
    canonical = (
        f"fingerprint={_pool_construction_fingerprint()}|families={families}|"
        f"instances={instances_per_family}|frames={frames}|sample_rate={sample_rate}|"
        f"jitter={jitter}|seed={seed}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


_POOL_CACHE_DONE_MARKER = "_DONE"


def _load_or_build_pool(
    pool_cache_dir: Path | None,
    families: int,
    instances_per_family: int,
    frames: int,
    sample_rate: int,
    jitter: int,
    seed: int,
) -> dict[str, bytes]:
    """[`_build_pool`] straight through when `pool_cache_dir` is `None`
    (real runs never set it -- see module doc's `--pool-cache-dir`) --
    IDENTICAL bytes to every invocation before this cache existed. With a
    cache dir given: a cache HIT reads every expected file straight off
    disk (never re-runs the waveform synthesis loop); a cache MISS builds
    the pool once via `_build_pool`, writes it into a fresh per-key
    subdirectory, and only then drops the `_DONE` marker that makes it
    visible to a later hit -- so a process that dies mid-write leaves an
    incomplete (marker-less) subdirectory that the NEXT invocation rebuilds
    from scratch, rather than one that reads back a partial pool.
    """
    if pool_cache_dir is None:
        return _build_pool(families, instances_per_family, frames, sample_rate, jitter, seed)

    key = _pool_cache_key(families, instances_per_family, frames, sample_rate, jitter, seed)
    cache_dir = Path(pool_cache_dir) / key
    marker = cache_dir / _POOL_CACHE_DONE_MARKER
    expected_names = [
        _clip_name(family, instance)
        for family in range(families)
        for instance in range(instances_per_family)
    ]
    if marker.is_file():
        return {name: (cache_dir / name).read_bytes() for name in expected_names}

    files = _build_pool(families, instances_per_family, frames, sample_rate, jitter, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (cache_dir / name).write_bytes(data)
    marker.write_text(
        f"families={families} instances={instances_per_family} frames={frames} "
        f"sample_rate={sample_rate} jitter={jitter} seed={seed}\n"
    )
    return files


def generate_corpus(
    rows: int,
    seconds: float,
    sample_rate: int,
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
        seconds=seconds,
        sample_rate=sample_rate,
        seed=seed,
        families=families,
        instances_per_family=instances_per_family,
        jitter=jitter,
    )
    return files, train_rows


def generate_split(
    rows: int,
    seconds: float,
    sample_rate: int,
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
    where `files` maps a relative file name to its WAV bytes, `rows` is the
    train JSONL row list, and `heldout_rows_list` is the held-out one
    (EMPTY unless `heldout_rows > 0`).

    Pure with respect to its arguments -- no filesystem, no clock, no
    environment -- so determinism is testable without writing anything --
    UNLESS `pool_cache_dir` is given (opt-in only; every real invocation
    leaves it `None`), in which case the audio POOL (never the row lists)
    is read from or written to that directory (see `_load_or_build_pool`)
    but the RETURNED bytes are, by construction, identical either way.
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

    # The family pool is SPLIT only when a held-out set is requested; with
    # `--heldout-rows 0` (the default) `train_families == families` and
    # every line below runs exactly as it did before this flag existed.
    train_families = families
    if heldout_rows > 0:
        train_families = validate_heldout_split(
            families, heldout_families, heldout_rows, heldout_batch
        )

    # Fixed draw order: family-major, instance-minor (family J).
    # `_build_pool` is the SAME function whether or not `pool_cache_dir`
    # is given (see `_load_or_build_pool`'s own doc) -- caching can only
    # skip re-running this construction, never change what it would have
    # produced.
    files = _load_or_build_pool(
        pool_cache_dir, families, instances_per_family, frames, sample_rate, jitter, seed
    )

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
    """Write the WAVs and the JSONL under `out_dir` (created if absent), in
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
        help="OPT-IN (default: unset). When given, the family x instances audio POOL -- a "
        "pure function of (--families, --instances-per-family, --seconds, --sample-rate, "
        "--jitter, --seed) that never depends on --rows/--heldout-*/--out-dir -- is read from "
        "(or, on a first call, written to) a per-key subdirectory of this path instead of "
        "being re-synthesized on every invocation. Emitted bytes are byte-identical with or "
        "without this flag (see test_gen_fixed_length_audio_corpus.py); real runs never set "
        "it -- it exists ONLY to let a test harness synthesize a shared pool once across many "
        "invocations of this producer at the same (families, instances, seconds, sample-rate, "
        "jitter, seed) tuple.",
    )
    args = ap.parse_args(argv)

    try:
        files, rows, heldout_rows = generate_split(
            rows=args.rows,
            seconds=args.seconds,
            sample_rate=args.sample_rate,
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
        print(f"::error::gen_fixed_length_audio_corpus: {e}", file=sys.stderr)
        return 2

    jsonl_path = write_corpus(files, rows, args.out_dir, args.jsonl_name, heldout_rows)
    frames = frame_count(args.seconds, args.sample_rate)
    print(
        f"gen_fixed_length_audio_corpus: wrote {len(files)} clips of {frames} frames "
        f"({args.seconds}s at {args.sample_rate} Hz, 16-bit mono PCM) and {len(rows)} triplet "
        f"rows to {jsonl_path} (families={args.families}, "
        f"instances_per_family={args.instances_per_family}, jitter={args.jitter}, "
        f"seed={args.seed})"
    )
    if heldout_rows:
        print(
            f"gen_fixed_length_audio_corpus: wrote {len(heldout_rows)} held-out rows to "
            f"{args.out_dir / _HELDOUT_IDS_NAME} + {args.out_dir / _HELDOUT_JSONL_NAME} "
            f"(heldout_families={args.heldout_families} reserved from the family pool, "
            f"heldout_batch={args.heldout_batch})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
