#!/usr/bin/env python3
"""Issue #421 tower-profile ARTIFACT step: turn `profile_421_merge.py`'s and
`profile_421_attribute.py`'s own reports into the single committed
`crates/jammi-kernels/artifacts/cuda-runs/*.json` `check_cuda_run_artifacts.py`
schema-checks (contract `scratchpad/contract-421-profile.md` v2.5
`## Artifacts`).

## Why a third script, not hand-assembled JSON

The merge and attribution reports already carry every NUMBER this artifact
needs (positive-proof witnesses, per-step wall/front/busy/residual, chain
shares, `UNATTRIBUTED`, realized gains, the candidate-port verdict strings).
Hand-typing the artifact JSON from those reports would be exactly the
"restored coverage 0.867 -> 0.895" transcription failure mode the house
principles name: a value could drift from its source, or never have come
from a real run at all, and the artifact would read as equally authoritative
either way. This module is the single reader that turns
`--merge-json` + `--attribution-json` + a small, non-numeric identity
sidecar (`--identity`; device strings, checkpoint repo/file names, prose —
see `profile_421_run2_identity.json`'s own header comment for exactly what
it is and is not allowed to carry) into the artifact, and is tested
hermetically against SYNTHETIC merge/attribution/identity/legs-dir fixtures
(`test_profile_421_artifact.py`) — no GPU, no pod, no committed baseline.

## What comes from where

- `git_sha` / `box`: read off every leg's own `manifest.json` under
  `--legs-dir` — WITNESSED per run, never declared in the identity sidecar
  — and cross-checked to agree on every single one. A run whose legs
  disagree on which build or which physical box produced them is not one
  measurement session; this module refuses rather than pick one
  arbitrarily. Whenever `--merge-json` itself carries any `p2_bf16` rows,
  `--p2-dir` is REQUIRED (never optional) and EVERY named P2 tower's own
  `manifest.json` must exist under it and agree too — a missing or
  disagreeing P2 manifest is a refusal, never a silent skip past that
  tower. The witnessed `(towers, git_sha, box)` triple is recorded at
  `p2_witnessed` (`None` when `--merge-json` carries no `p2_bf16` rows at
  all).
- `status`: `"GREEN"` iff `--merge-json`'s own `summary.legs_invalid == 0`
  and `summary.p2_fail == 0`, else a `"RED: ..."` string naming the actual
  counts — derived, never hand-declared by the identity sidecar (see
  `derive_status`).
- `checkpoint_weights_sha256` (per tower family): read off `--merge-json`'s
  own per-leg `checkpoint_weights_sha256` (already cross-checked equal
  within a tower by `profile_421_merge.py`'s own `_check_cross_tower_
  identity`) — cross-checked AGAIN here across every leg of a family before
  it is trusted as "the" sha256 for that family's `notes.checkpoints` entry.
  A leg whose tower HAS a known checkpoint family but no sha at all is a
  refusal, never a silently-skipped leg.
- `launches_per_step` (used only by the CLIP launch-bound finding below):
  read off each named leg's own `census.json` under `--legs-dir` — the one
  number neither the merge nor the attribution report carries.
- Every per-leg wall/front/busy/residual number, every chain share, every
  realized-gain delta, every candidate-port verdict string: copied VERBATIM
  from `--merge-json` / `--attribution-json` — the verdict strings in
  `candidate_decisions` are copied character-for-character, never
  reformatted or re-derived, per the contract's own "the verdict strings
  written once (never edited)" line.
- `limits`: `attribution_report["limits"]`, read by name and cross-checked
  to agree with the LIVE `profile_421_attribute.py` constants of the same
  name (`_resolve_recorded_limits`) — the run's own recorded validity-gate
  bounds, never a value re-parsed out of a leg's own `decision_grade_
  reason` string and never the live import trusted directly. Every
  downstream reader of a bound (this module's own identity-sidecar
  clauses, `profile_421_render_docs.py`'s `render_htsat_a2_deviation`)
  reads it from here.
- `notes.recorded_deviations`: `--identity`'s own entries are TEMPLATES —
  any number the prose needs to quote (a chain share, a validity-gate
  bound, a corpus file/row count) is a `{placeholder}` this module fills
  from a live source (`_identity_template_context`) — see
  `profile_421_run2_identity.json`'s own header comment for the "no bare
  digit" guarantee this enforces. A run fact with NO in-tree witness at all
  (e.g. an off-pod disk-usage total) belongs in `notes.operator_recorded`
  instead, explicitly outside that guarantee.
- Everything else in `notes` (`what`, `gpu`, `driver`, `cpu`, `nsys`,
  checkpoint repo/file names, the producer `invocation` string): read from
  `--identity`, which carries ONLY provenance with no numeric guarantee
  attached (a device-model string is not a measurement an oracle could
  re-derive) — see that file's own header comment.

## Findings: computed, not hand-typed

`findings` (the four items the contract's own close-out names: HTSAT
front-end share, CLIP-vision front-end share, CLIP launch-bound + the BF16
busy/wall trade at batch 8, and the out-of-tier `C-ATTN-HTSAT` number) are
built by `compute_findings()` below from the SAME merge/attribution numbers
already read for `legs`/`attribution` — the prose sentence is an f-string
formatted around values this module computed, not a string a human typed a
percentage into. Each finding also carries its own `evidence` block (the
raw leg ids and the exact numeric inputs the sentence was built from), so a
downstream reader (the docs-ci guide table, an auditor) can re-derive the
rounded number in the sentence from the unrounded evidence without trusting
the sentence's own English. The CLIP launch-bound finding's own BF16-vs-F32
wording is SIGN-DERIVED, never a hard-coded "cuts"/"drops": a blanket verb
is used only when every tower's delta agrees on direction, else each tower
gets its own sign-derived clause (`_direction_clause`, with its own shape
for an exact-zero delta). Every finding is computed ONLY from legs whose
`--merge-json` verdict is VALID (and, for a finding built off a CHAIN
SHARE, whose `--attribution-json` row is also `decision_grade`) — a finding
whose required leg(s) fail either gate is never built; it is instead named,
with its leg(s) and the reason, in the sibling `suppressed_findings` list
(`compute_findings`'s own `_merge_verdict_problems`/
`_decision_grade_problems`).

Every QUALITATIVE word a finding's own prose can use ("front-end-bound",
"dtype- and arm-invariant", "launch-bound", "only") is gated behind a
NAMED, numeric rule (`FRONT_END_BOUND_SHARE_OF_WALL_MIN`,
`ARM_INVARIANCE_REL_SPREAD_MAX`, `LAUNCH_BOUND_LAUNCHES_PER_STEP_MIN` +
`LAUNCH_BOUND_RESIDUAL_SHARE_OF_WALL_MIN`, and the "only" pointwise
magnitude comparison), evaluated against the SAME required legs and
recorded — by name, threshold, and outcome — in that finding's own
`evidence` block. When a rule does not hold on every leg it was evaluated
against, the finding still builds (never suppressed for a wording reason
alone) but drops the word for a neutral sentence stating the same numbers
— a leg going from decisively front-end-bound to marginally so must change
the SENTENCE, not just the number inside it, and a reader must be able to
see the exact bar that either cleared or did not from `evidence` alone,
never from re-deriving the English.

## Producer identity (regeneration provenance)

The CLI (`main`, not the hermetic `build_report` core) stamps
`producer.identity` (the fixed marker `"source_sha256+input_manifest"`,
checked by `check_cuda_run_artifacts.py`'s rule (j) — a producer that
stamps this marker MUST carry both blocks below, never just one),
`producer.source_sha256` (the sha256 of every file in
`SOURCE_FILES_FOR_NUMBERS` below, keyed by its repo-root-relative posix
path), `producer.input_sha256` (the sha256 of the `--merge-json`/
`--attribution-json`/`--identity` FILES as given, PLUS a full manifest over
every file this module itself reads off `--legs-dir`/`--p2-dir` — see
"Input-completeness" below), and `producer.invocation_argv` (this run's own
argv) onto the artifact.

`source_sha256` replaces a git commit sha (the old `producer.tree_sha`,
`JAMMI_BUILD_SHA` if set else `git rev-parse HEAD`). A commit sha is the
wrong determinant here: this artifact cannot know, at render time, which
future commit will contain it (an ordinary `git commit` of this very file
changes `HEAD` out from under an already-rendered JSON with no code
change at all), and nothing ever validated a dirty-tree render's `tree_sha`
against anything — it was stamped and then never re-checked. The
producer's own CONTENT identity is the right thing to check regeneration
against instead: `check_cuda_run_artifacts.py` recomputes the sha256 of
every path named in `source_sha256` at ITS OWN HEAD and refuses a mismatch
BY NAME, so editing this module (or a file it reads a live constant from —
`profile_421_attribute.py`'s validity-gate constants,
`profile_421_legs.sh`'s media-corpus shell constants, the two corpus
producers' own `_DEFAULT_INSTANCES_PER_FAMILY`, `test_profile_421_merge.py`'s
own hermetic-suite size) and forgetting to regenerate the committed artifact
is now a hard CI failure, never a silent staleness. Regenerating from the
SAME `--legs-dir`/`--p2-dir` tree (byte-identical to the one the input
manifest below names) and the SAME three top-level report files against the
SAME producer source bytes reproduces a byte-identical artifact (proof by
regeneration: `test_profile_421_artifact.py`'s own
`RealFixtureRegenerationTests` regenerates the committed artifact from the
committed `ci/scripts/perf/fixtures/profile_421_run2/` fixture and diffs it
byte-for-byte before every commit that touches a source file or that
fixture); regenerating after editing this module's wording (even prose-only)
changes `source_sha256` (and, if a finding's wording changed, the affected
`findings[].text`) while every measured NUMBER stays the same — the two
kinds of change are always distinguishable from the diff alone.

### Input-completeness

`producer.input_sha256` used to name only the three TOP-LEVEL report files
(`--merge-json`/`--attribution-json`/`--identity`) — leaving every byte this
module ALSO reads directly off `--legs-dir`/`--p2-dir` (every leg's own
`manifest.json` and `census.json`, plus `census.pre-demangle.json` for
every leg in `KERNEL_IDENTITY_SPLIT_LEGS`, plus every witnessed P2 tower's
own `manifest.json`) output-affecting but uncaptured. `build_input_manifest`
walks exactly that closed, declared file
set (`LEG_INPUT_FILENAMES`, `KERNEL_IDENTITY_SPLIT_LEGS`,
`P2_TOWER_INPUT_FILENAMES` below) and hashes every one of those files,
keyed `"legs/<leg_id>/<filename>"` / `"p2/<tower>/<filename>"` (a
LEG/TOWER-RELATIVE key, never an absolute path, so the manifest's own
VALUES reproduce byte-identically regardless of which directory
`--legs-dir`/`--p2-dir` happen to be mounted at — a pod's own scratch path
vs. this repo's own committed fixture directory). Every actual read this
module performs off `--legs-dir`/`--p2-dir` is routed through `_leg_file`/
`_p2_tower_file`, which refuse (by name) a filename outside that declared
set — a future edit that starts reading some new per-leg file without
first adding it to `LEG_INPUT_FILENAMES` (and therefore to the manifest) is
a hard refusal at the point of the read, never a silently uncaptured byte.
`run_n.json`/`run_m.json` (per leg) and each P2 tower's own `run.json` are
NOT in this module's own manifest — `profile_421_merge.py` reads those to
produce `--merge-json`, whose bytes are already covered by
`input_sha256["merge_json"]`; they, and every other file the merge/
attribution pipeline reads, are committed in the fixture directory (for the
regeneration test's own end-to-end proof) but are not this module's OWN
declared read set.

Run: `python3 ci/scripts/perf/profile_421_artifact.py --legs-dir <dir>
[--p2-dir <dir>] --merge-json <path> --attribution-json <path>
--identity <path> [--out <path>]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unittest
from pathlib import Path

# `profile_421_attribute.py`'s own validity-gate constants — imported, never
# retyped, so a refusal can name the LIVE `UNATTRIBUTED_DECISION_GRADE_LIMIT`/
# `UNKNOWN_KERNEL_SHARE_LIMIT` this repo's attribution logic currently
# enforces when it disagrees with the bound of the SAME name a run's own
# `--attribution-json` recorded in its own top-level `limits` block
# (`profile_421_attribute.py`'s module doc, "`limits`: the report's own
# recorded validity-gate bounds"). `_recorded_limit` below never quotes this
# import directly for a PRINTED bound — every bound this module prints or
# compares against is read from `attribution_report["limits"]` by name; the
# live constant is consulted ONLY to refuse a regeneration whose value has
# since moved away from what the run actually recorded, naming both (see
# `_recorded_limit`'s own doc). Never a leg's own `decision_grade_reason`
# string either — a leg that failed for an unrelated reason (or never
# failed at all) records no such string to parse in the first place.
import profile_421_attribute as _attribute_mod

# `test_profile_421_merge.py`'s own hermetic-suite SIZE — imported (never
# executed: `unittest.defaultTestLoader.loadTestsFromModule` enumerates test
# methods without ever calling `.run()`) so the identity sidecar's "the live
# suite already carried N tests" prose quotes a live `unittest` discovery
# count over the REAL committed test file, never a hand-typed number that
# could silently drift as tests are added or removed (see
# `compute_merge_suite_test_count` below). Its own bytes are therefore one of
# this module's `SOURCE_FILES_FOR_NUMBERS` too.
import test_profile_421_merge as _merge_test_mod

SCHEMA_VERSION = 1

PERF_DIR = Path(__file__).resolve().parent
# `ci/scripts/perf/profile_421_artifact.py` -> repo root is three parents up
# (`ci/scripts/perf` -> `ci/scripts` -> `ci` -> repo root).
REPO_ROOT = PERF_DIR.parents[2]
LEGS_SH = PERF_DIR / "profile_421_legs.sh"
IMAGE_CORPUS_PY = PERF_DIR / "gen_fixed_shape_image_corpus.py"
AUDIO_CORPUS_PY = PERF_DIR / "gen_fixed_length_audio_corpus.py"
ATTRIBUTE_MODULE_PY = PERF_DIR / "profile_421_attribute.py"
MERGE_TEST_PY = PERF_DIR / "test_profile_421_merge.py"

# Every file whose BYTES can change a NUMBER (never just prose) this producer
# emits: itself, the attribution module it imports for the validity-gate
# constants `_identity_template_context` quotes, the three driver/producer
# files `compute_media_corpus_pool` regexes live constants out of, and the
# test module `compute_merge_suite_test_count` counts test cases from.
# Deliberately NOT `profile_421_merge.py`: this module imports no constant
# from it and reads no number off its own source (`--merge-json` is a
# separate FILE input already covered by `producer.input_sha256`, not a
# module import) — listing it here would assert a coupling this producer
# does not actually have.
SOURCE_FILES_FOR_NUMBERS: tuple[Path, ...] = (
    Path(__file__).resolve(),
    ATTRIBUTE_MODULE_PY,
    LEGS_SH,
    IMAGE_CORPUS_PY,
    AUDIO_CORPUS_PY,
    MERGE_TEST_PY,
)


class ArtifactBuildError(Exception):
    """Uncomputable or inconsistent input — fails closed, never guesses."""


# The fixed marker `producer.identity` is stamped with — `check_cuda_run_
# artifacts.py`'s rule (j) treats this EXACT string as "this producer
# declares BOTH `source_sha256` and `input_sha256` always present together"
# (see that module's own `PRODUCER_SOURCE_IDENTITY_MARKER` and
# `SOURCE_IDENTITY_DECLARING_PRODUCER_PATHS`, which names
# `ci/scripts/perf/profile_421_legs.sh` — this module's own `producer.path`
# — as a known convention-declarer that must stamp this marker on every
# artifact it emits).
PRODUCER_SOURCE_IDENTITY_MARKER = "source_sha256+input_manifest"


def _finite(value: object, label: str) -> float:
    """A float read that refuses non-finite input AT THE POINT OF READING —
    `profile_421_merge.py`'s own `finite_float` doctrine, restated here
    because every finding below does ARITHMETIC (subtraction, division,
    min/max) on values read out of `--merge-json`/`--attribution-json`/
    `census.json`. An ordinary threshold or `min`/`max` call would let a
    diverged (`NaN`) upstream measurement sail through silently: `NaN > 0`,
    `NaN < 0` and `NaN == 0` are ALL `False`, and `min(nan, 1.0)` is
    order-dependent, not a refusal. Every value this module ever compares or
    divides is routed through here first."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactBuildError(f"{label} is not a number ({value!r})")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ArtifactBuildError(f"{label} is not finite ({value!r})")
    return as_float


def _finite_share(value: object, label: str) -> float:
    """`_finite` plus a `[0, 1]` domain check — every value this module
    multiplies by `100.0` to report as a percentage is a SHARE (of wall or
    GPU-busy time), and a share outside `[0, 1]` (negative, or > 1 from a
    mis-summed denominator upstream) is exactly as uncomputable as a `NaN`:
    it is not a number this module should silently round and print. Kept
    separate from `_finite` (which is also used for plain durations and
    counts that are never bounded to `[0, 1]`)."""
    as_float = _finite(value, label)
    if not (0.0 <= as_float <= 1.0):
        raise ArtifactBuildError(f"{label} is not in [0, 1] ({as_float!r})")
    return as_float


def _load_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {what} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactBuildError(f"{what} at {path} is not valid JSON: {exc}") from exc


def _load_text(path: Path, what: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {what} at {path}: {exc}") from exc


def _sha256_file(path: Path, what: str) -> str:
    """The sha256 of an INPUT FILE's own bytes (`--merge-json`,
    `--attribution-json`, `--identity`) — recorded in the artifact's own
    `producer` block so a reader can confirm this exact JSON was rendered
    from THESE exact input files, never a hand-typed or assumed identity."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {what} at {path} to hash it: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def _source_sha256() -> dict[str, str]:
    """The sha256 of every file in `SOURCE_FILES_FOR_NUMBERS`, keyed by its
    repo-root-relative posix path — this producer's own CONTENT identity,
    replacing a git commit sha (the old `tree_sha`) as the thing a reader
    checks regeneration against (see the module doc's "Producer identity"
    section for why a commit sha was the wrong determinant). Refuses rather
    than guess: a file this module depends on for a NUMBER that cannot be
    read at all is exactly as unverifiable as a producer block that never
    recorded its own identity."""
    out: dict[str, str] = {}
    for path in SOURCE_FILES_FOR_NUMBERS:
        resolved = path.resolve()
        try:
            data = resolved.read_bytes()
        except OSError as exc:
            raise ArtifactBuildError(f"could not read producer source file {resolved} to hash it: {exc}") from exc
        try:
            rel = resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise ArtifactBuildError(f"producer source file {resolved} is not under REPO_ROOT {REPO_ROOT}") from exc
        out[rel] = hashlib.sha256(data).hexdigest()
    return out


# --------------------------------------------------------------------------- #
# Driver/producer constants read live off their OWN source files — the media-
# corpus pool arithmetic ("families x instances = N files, M distinct train
# clips") the identity sidecar's own deviation prose quotes has no other
# witness anywhere in a committed report, so it is read here from
# `profile_421_legs.sh`'s own shell constants and the two corpus producers'
# own `_DEFAULT_INSTANCES_PER_FAMILY`, never hand-retyped as a second,
# independently-drifting copy.
# --------------------------------------------------------------------------- #
_SHELL_INT_CONST_RE = r'(?m)^{name}=(?:"\$\{{[A-Z0-9_]+:-(\d+)\}}"|(\d+)\b)'
_PY_INT_CONST_RE = r"(?m)^{name}\s*=\s*(\d+)\b"


def _read_shell_int_const(path: Path, name: str) -> int:
    text = _load_text(path, f"shell script (for constant {name!r})")
    m = re.search(_SHELL_INT_CONST_RE.format(name=re.escape(name)), text)
    if not m:
        raise ArtifactBuildError(f"could not find shell constant {name!r} in {path}")
    raw = m.group(1) if m.group(1) is not None else m.group(2)
    return int(raw)


def _read_py_int_const(path: Path, name: str) -> int:
    text = _load_text(path, f"python module (for constant {name!r})")
    m = re.search(_PY_INT_CONST_RE.format(name=re.escape(name)), text)
    if not m:
        raise ArtifactBuildError(f"could not find Python constant {name!r} in {path}")
    return int(m.group(1))


def compute_media_corpus_pool() -> dict[str, int]:
    """The media-corpus pool numbers the identity sidecar's own deviation
    prose quotes ("families x instances = N files ... M distinct train
    clips", "the M-leg's R rows"), read live off `profile_421_legs.sh`'s own
    `MEDIA_FAMILIES`/`MEDIA_HELDOUT_FAMILIES`/`STEPS_M`/`BATCH` and the two
    corpus producers' own `_DEFAULT_INSTANCES_PER_FAMILY` — cross-checked to
    agree with each other (image vs audio) before either is trusted."""
    media_families = _read_shell_int_const(LEGS_SH, "MEDIA_FAMILIES")
    media_heldout_families = _read_shell_int_const(LEGS_SH, "MEDIA_HELDOUT_FAMILIES")
    steps_m = _read_shell_int_const(LEGS_SH, "STEPS_M")
    batch = _read_shell_int_const(LEGS_SH, "BATCH")
    image_instances = _read_py_int_const(IMAGE_CORPUS_PY, "_DEFAULT_INSTANCES_PER_FAMILY")
    audio_instances = _read_py_int_const(AUDIO_CORPUS_PY, "_DEFAULT_INSTANCES_PER_FAMILY")
    if image_instances != audio_instances:
        raise ArtifactBuildError(
            "image/audio media corpus producers disagree on _DEFAULT_INSTANCES_PER_FAMILY: "
            f"{image_instances} vs {audio_instances}"
        )
    if media_heldout_families > media_families:
        raise ArtifactBuildError(
            f"profile_421_legs.sh: MEDIA_HELDOUT_FAMILIES ({media_heldout_families}) exceeds "
            f"MEDIA_FAMILIES ({media_families})"
        )
    return {
        "corpus_total_files": media_families * image_instances,
        "corpus_train_clips": (media_families - media_heldout_families) * image_instances,
        "corpus_rows_m": batch * steps_m,
    }


def compute_merge_suite_test_count() -> int:
    """`profile_421_merge.py`'s own hermetic suite's LIVE test count — the
    identity sidecar's own "the live suite already carried N tests"
    deviation quotes this, via `unittest`'s own discovery
    (`TestLoader.loadTestsFromModule(...).countTestCases()`, which
    ENUMERATES test methods without RUNNING any of them) against the REAL
    committed `test_profile_421_merge.py`, never a hand-typed count that
    could silently drift as tests are added or removed."""
    return unittest.defaultTestLoader.loadTestsFromModule(_merge_test_mod).countTestCases()


def compute_sqlite_raw_export_count(legs_dir: Path) -> int:
    """Two raw `.sqlite` exports (`run_n.sqlite`, `run_m.sqlite`) per leg —
    the count the identity sidecar's own "the raw sqlite exports (N files,
    ...)" deviation quotes, read live off how many legs `--legs-dir` itself
    actually carries, never a hand-typed literal."""
    return len(_leg_dirs(legs_dir)) * 2


def _leg_dirs(legs_dir: Path) -> list[Path]:
    if not legs_dir.is_dir():
        raise ArtifactBuildError(f"--legs-dir {legs_dir} is not a directory")
    return sorted((p for p in legs_dir.iterdir() if p.is_dir()), key=lambda p: p.name)


# --------------------------------------------------------------------------- #
# input-completeness: the CLOSED set of filenames this
# module itself reads off `--legs-dir`/`--p2-dir`, and the ONE gateway every
# such read goes through — a filename outside this set is refused AT THE
# READ, never silently uncaptured by `producer.input_sha256` (see the module
# doc's "Input-completeness" section).
# --------------------------------------------------------------------------- #
LEG_INPUT_FILENAMES: tuple[str, ...] = ("manifest.json", "census.json")
# Legs whose `census.pre-demangle.json` this module ALSO reads (via
# `compute_kernel_identity_split_count`, called once per leg named here from
# `_identity_template_context`) — kept as an explicit, closed list (rather
# than "whichever leg the call site happens to name") so `build_input_
# manifest` below can name exactly these bytes without first running
# `_identity_template_context` to discover them.
KERNEL_IDENTITY_SPLIT_LEGS: tuple[str, ...] = ("clip-text-A2",)
P2_TOWER_INPUT_FILENAMES: tuple[str, ...] = ("manifest.json",)


def _leg_file(legs_dir: Path, leg_id: str, filename: str, what: str) -> dict:
    """The ONE gateway for reading a per-leg input file under `--legs-dir`.
    `filename` outside `LEG_INPUT_FILENAMES` (or `census.pre-demangle.json`
    for a leg outside `KERNEL_IDENTITY_SPLIT_LEGS`) is refused HERE, at the
    point of the read — a byte this producer starts reading without first
    declaring it in the closed set above would otherwise be an uncaptured
    input: output-affecting, but invisible to `producer.input_sha256`."""
    allowed = filename in LEG_INPUT_FILENAMES or (
        filename == "census.pre-demangle.json" and leg_id in KERNEL_IDENTITY_SPLIT_LEGS
    )
    if not allowed:
        raise ArtifactBuildError(
            f"{leg_id}: refusing to read {filename!r} — not in LEG_INPUT_FILENAMES/"
            "KERNEL_IDENTITY_SPLIT_LEGS; declare it there (so build_input_manifest captures it) "
            "before reading it"
        )
    return _load_json(legs_dir / leg_id / filename, what)


def _p2_tower_file(p2_dir: Path, tower: str, filename: str, what: str) -> dict:
    """The ONE gateway for reading a per-P2-tower input file under
    `--p2-dir` — mirrors `_leg_file` above; a filename outside
    `P2_TOWER_INPUT_FILENAMES` is refused at the read."""
    if filename not in P2_TOWER_INPUT_FILENAMES:
        raise ArtifactBuildError(
            f"p2/{tower}: refusing to read {filename!r} — not in P2_TOWER_INPUT_FILENAMES; declare it "
            "there (so build_input_manifest captures it) before reading it"
        )
    return _load_json(p2_dir / tower / filename, what)


def build_input_manifest(legs_dir: Path, p2_dir: Path | None, p2_towers: list[str]) -> dict[str, str]:
    """Walks the CLOSED, declared file set (`LEG_INPUT_FILENAMES`,
    `KERNEL_IDENTITY_SPLIT_LEGS`, `P2_TOWER_INPUT_FILENAMES`) and hashes
    every one of those files — this producer's own full account of every
    byte it reads off `--legs-dir`/`--p2-dir`, keyed `"legs/<leg_id>/
    <filename>"` / `"p2/<tower>/<filename>"` (a leg/tower-RELATIVE key,
    never an absolute path, so the manifest's own VALUES reproduce byte-
    identically regardless of which directory `--legs-dir`/`--p2-dir`
    happen to be mounted at). A declared file that does not exist is a hard
    refusal, never a silently-shrunk manifest."""
    manifest: dict[str, str] = {}
    for leg_dir in _leg_dirs(legs_dir):
        for filename in LEG_INPUT_FILENAMES:
            path = leg_dir / filename
            if not path.is_file():
                raise ArtifactBuildError(f"{leg_dir.name}: expected input file {filename!r} missing under --legs-dir")
            manifest[f"legs/{leg_dir.name}/{filename}"] = _sha256_file(path, f"{leg_dir.name}/{filename}")
    for leg_id in KERNEL_IDENTITY_SPLIT_LEGS:
        path = legs_dir / leg_id / "census.pre-demangle.json"
        if not path.is_file():
            raise ArtifactBuildError(f"{leg_id}: expected input file 'census.pre-demangle.json' missing under --legs-dir")
        manifest[f"legs/{leg_id}/census.pre-demangle.json"] = _sha256_file(path, f"{leg_id}/census.pre-demangle.json")
    if p2_towers:
        if p2_dir is None:
            raise ArtifactBuildError("--merge-json carries p2_bf16 rows but --p2-dir is None building the input manifest")
        for tower in p2_towers:
            for filename in P2_TOWER_INPUT_FILENAMES:
                path = p2_dir / tower / filename
                if not path.is_file():
                    raise ArtifactBuildError(f"p2/{tower}: expected input file {filename!r} missing under --p2-dir")
                manifest[f"p2/{tower}/{filename}"] = _sha256_file(path, f"p2/{tower}/{filename}")
    return manifest


def p2_tower_names(merge_report: dict) -> list[str]:
    """The distinct, sorted tower names `--merge-json`'s own `p2_bf16` rows
    name — the set of P2 towers THIS merge report claims to have measured,
    and therefore the exact set `collect_identity` below must witness a
    manifest for. Never a directory listing (a tower this run never
    measured has no row here at all, and a `p2_dir` scan could otherwise
    pick up a stale or unrelated subdirectory).

    A `p2_bf16` row with no non-empty string `tower` is a hard refusal,
    never a silent drop — a dropped row could otherwise shrink this set (in
    the limit, to empty) even though `--merge-json` genuinely carries
    `p2_bf16` rows, which would make `p2_witnessed: null` downstream lie
    about "no p2_bf16 rows in the merge" and (worse) let `--p2-dir` become
    silently optional for a merge report that DID measure BF16 legs."""
    towers: set[str] = set()
    for i, row in enumerate(merge_report.get("p2_bf16", [])):
        if not isinstance(row, dict):
            raise ArtifactBuildError(f"--merge-json p2_bf16[{i}] is not an object ({row!r})")
        tower = row.get("tower")
        if not isinstance(tower, str) or not tower:
            raise ArtifactBuildError(f"--merge-json p2_bf16[{i}] has no non-empty string tower ({tower!r})")
        towers.add(tower)
    return sorted(towers)


# --------------------------------------------------------------------------- #
# git_sha / box: witnessed off every manifest.json, cross-checked to agree
# --------------------------------------------------------------------------- #
def collect_identity(legs_dir: Path, p2_dir: Path | None, p2_towers: list[str]) -> tuple[str, str, list[str]]:
    """Reads `git_sha` + `box` off every leg's own `manifest.json`, and — if
    `p2_towers` (the towers `--merge-json`'s own `p2_bf16` rows name) is
    non-empty — off every ONE of those P2 towers' own `manifest.json` under
    `--p2-dir` too, refusing (`ArtifactBuildError`) unless every single one
    agrees on `(git_sha, box)`: a mixed-build or mixed-box session is not one
    measurement, and this module never silently picks the first value it
    saw.

    When `p2_towers` is non-empty, `--p2-dir` is REQUIRED (a merge report
    that measured P2 towers but was handed no `--p2-dir` to witness them
    from is exactly the "byte-identical without --p2-dir" gap this function
    closes), and a tower named in `p2_towers` with no `manifest.json` under
    `--p2-dir` is a hard refusal, never a silent `continue` past it.

    Returns `(git_sha, box, p2_towers)` — the third element is `p2_towers`
    itself, echoed back so the caller can build the artifact's own
    `p2_witnessed` block without re-deriving it.
    """
    seen: dict[str, tuple[str, str]] = {}
    for leg_dir in _leg_dirs(legs_dir):
        manifest = _leg_file(legs_dir, leg_dir.name, "manifest.json", f"{leg_dir.name}: manifest.json")
        git_sha = manifest.get("git_sha")
        box = manifest.get("box")
        if not isinstance(git_sha, str) or not git_sha:
            raise ArtifactBuildError(f"{leg_dir.name}: manifest.json has no string git_sha")
        if not isinstance(box, str) or not box:
            raise ArtifactBuildError(f"{leg_dir.name}: manifest.json has no string box")
        seen[f"leg:{leg_dir.name}"] = (git_sha, box)
    if p2_towers:
        if p2_dir is None:
            raise ArtifactBuildError(
                "--merge-json carries p2_bf16 rows for "
                f"{p2_towers} but --p2-dir was not given — every P2 tower's own manifest.json "
                "must be witnessed, never assumed to agree"
            )
        if not p2_dir.is_dir():
            raise ArtifactBuildError(f"--p2-dir {p2_dir} is not a directory")
        for tower in p2_towers:
            manifest_path = p2_dir / tower / "manifest.json"
            if not manifest_path.is_file():
                raise ArtifactBuildError(
                    f"p2/{tower}: --merge-json's p2_bf16 names this tower but no manifest.json exists "
                    f"under --p2-dir {p2_dir} — a missing P2 manifest is a refusal, never a silent skip"
                )
            manifest = _p2_tower_file(p2_dir, tower, "manifest.json", f"p2/{tower}: manifest.json")
            git_sha = manifest.get("git_sha")
            box = manifest.get("box")
            if not isinstance(git_sha, str) or not git_sha:
                raise ArtifactBuildError(f"p2/{tower}: manifest.json has no string git_sha")
            if not isinstance(box, str) or not box:
                raise ArtifactBuildError(f"p2/{tower}: manifest.json has no string box")
            seen[f"p2:{tower}"] = (git_sha, box)
    if not seen:
        raise ArtifactBuildError("no manifest.json found under --legs-dir (or --p2-dir) to witness git_sha/box from")
    distinct = set(seen.values())
    if len(distinct) != 1:
        detail = "; ".join(f"{k}={v}" for k, v in sorted(seen.items()))
        raise ArtifactBuildError(f"manifests disagree on (git_sha, box) — not one measurement session: {detail}")
    (git_sha, box) = next(iter(distinct))
    return git_sha, box, p2_towers


# --------------------------------------------------------------------------- #
# checkpoint sha256 per tower family — cross-checked (again) across legs
# --------------------------------------------------------------------------- #
_TOWER_FAMILY = {
    "clip-text": "clip",
    "clip-vision": "clip",
    "htsat": "clap",
}


def collect_checkpoint_sha256(merge_legs: list[dict]) -> dict[str, str]:
    """A leg whose tower maps to a known checkpoint family (`_TOWER_FAMILY`)
    but carries no `checkpoint_weights_sha256` is a hard refusal, never a
    silent skip — a silently-skipped leg could hide behind its OWN family's
    other legs agreeing, reporting a checkpoint identity this leg itself
    never actually confirmed. This contract names exactly three towers
    (`_TOWER_FAMILY`'s own keys) and every one of them HAS a checkpoint
    family, so a leg whose tower is not in `_TOWER_FAMILY` at all is not a
    legitimate "no checkpoint" case — it is an unrecognized tower (a typo,
    or a tower this contract never declared) and is refused by name, never
    silently skipped past."""
    by_family: dict[str, set[str]] = {}
    for leg in merge_legs:
        tower = leg.get("tower")
        leg_id = leg.get("leg_id", "<unknown>")
        family = _TOWER_FAMILY.get(tower)
        if family is None:
            raise ArtifactBuildError(
                f"{leg_id}: tower {tower!r} is not one of this contract's known towers "
                f"({sorted(_TOWER_FAMILY)}) — an unrecognized tower is refused, never silently skipped"
            )
        sha = leg.get("checkpoint_weights_sha256")
        if not isinstance(sha, str) or not sha:
            raise ArtifactBuildError(
                f"{leg_id}: tower {tower!r} (checkpoint family {family!r}) has no "
                "checkpoint_weights_sha256 in --merge-json"
            )
        by_family.setdefault(family, set()).add(sha)
    out: dict[str, str] = {}
    for family, shas in by_family.items():
        if len(shas) != 1:
            raise ArtifactBuildError(
                f"checkpoint family {family!r} has disagreeing checkpoint_weights_sha256 across legs: {sorted(shas)}"
            )
        out[family] = next(iter(shas))
    return out


# --------------------------------------------------------------------------- #
# legs — merge + attribution + census launches_per_step, combined per leg id
# --------------------------------------------------------------------------- #
LEG_FIELDS_FROM_MERGE = (
    "tower",
    "task",
    "dtype",
    "arm",
    "kernels_disabled",
    "positive_proof",
    "per_step",
    "checkpoint_weights_sha256",
    "fusible_site_census",
)
LEG_FIELDS_FROM_ATTRIBUTION = (
    "signatures",
    "chains",
    "unknown_kernels",
    "decision_grade",
    "decision_grade_reason",
    "outside_signature_plausibly_attention",
    # HTSAT-only diagnostic (module doc, "Known, DOCUMENTED, UNRESOLVED
    # collision"); `.get()` below is `None` on every non-HTSAT leg, copied
    # verbatim rather than omitted so a consumer can tell "not applicable"
    # (`None`) apart from "measured zero".
    "ambiguous_out_mlp_collision",
)


def _index_by_leg_id(rows: list[dict]) -> dict[str, dict]:
    """Refuses a duplicate `leg_id` rather than let a plain dict comprehension
    silently keep only the LAST row with that id — a report that (by a
    producer bug or a hand-edit) carries two rows for the same leg would
    otherwise have one of them vanish with no signal at all."""
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "leg_id" not in row:
            continue
        leg_id = row["leg_id"]
        if leg_id in out:
            raise ArtifactBuildError(f"duplicate leg_id {leg_id!r} in the same report")
        out[leg_id] = row
    return out


def build_legs(merge_report: dict, attribution_report: dict, legs_dir: Path) -> list[dict]:
    merge_by_id = _index_by_leg_id(merge_report.get("legs", []))
    attr_by_id = _index_by_leg_id(attribution_report.get("legs", []))
    legs_dir_ids = {p.name for p in _leg_dirs(legs_dir)}

    merge_ids, attr_ids = set(merge_by_id), set(attr_by_id)
    if merge_ids != legs_dir_ids:
        raise ArtifactBuildError(
            "--merge-json's leg set does not match --legs-dir's leg set: "
            f"only in merge={sorted(merge_ids - legs_dir_ids)}, only in legs-dir={sorted(legs_dir_ids - merge_ids)}"
        )
    if attr_ids != legs_dir_ids:
        raise ArtifactBuildError(
            "--attribution-json's leg set does not match --legs-dir's leg set: "
            f"only in attribution={sorted(attr_ids - legs_dir_ids)}, only in legs-dir={sorted(legs_dir_ids - attr_ids)}"
        )

    legs: list[dict] = []
    for leg_dir in _leg_dirs(legs_dir):
        leg_id = leg_dir.name
        merge_leg = merge_by_id[leg_id]
        attr_leg = attr_by_id[leg_id]
        census = _leg_file(legs_dir, leg_id, "census.json", f"{leg_id}: census.json")
        launches = _finite(census.get("launches_per_step"), f"{leg_id}: census.json launches_per_step")

        row: dict[str, object] = {"leg_id": leg_id}
        for field in LEG_FIELDS_FROM_MERGE:
            row[field] = merge_leg.get(field)
        row["merge_verdict"] = merge_leg.get("verdict")
        row["merge_reasons"] = merge_leg.get("reasons")
        for field in LEG_FIELDS_FROM_ATTRIBUTION:
            row[field] = attr_leg.get(field)
        row["attribution_verdict"] = attr_leg.get("verdict")
        row["attribution_reasons"] = attr_leg.get("reasons")
        row["launches_per_step"] = launches
        problems = _contract_validity_problems(merge_by_id, attr_by_id, (leg_id,))
        row["contract_valid"] = not problems
        row["contract_validity_problems"] = problems
        legs.append(row)
    return legs


def build_contract_validity(merge_report: dict, attribution_report: dict, legs_dir: Path) -> dict:
    """The contract's validity gate tallied over every leg — the ONE
    headline count a downstream reader may quote ("N of M legs pass the
    contract's validity gate"), never the merge-level `status` alone, which
    reads only the gate's first three clauses."""
    merge_by_id = _index_by_leg_id(merge_report.get("legs", []))
    attr_by_id = _index_by_leg_id(attribution_report.get("legs", []))
    leg_ids = [p.name for p in _leg_dirs(legs_dir)]
    failing: list[dict] = []
    for leg_id in leg_ids:
        problems = _contract_validity_problems(merge_by_id, attr_by_id, (leg_id,))
        if problems:
            failing.append(
                {
                    "leg_id": leg_id,
                    "problems": problems,
                    "decision_grade_reason": attr_by_id[leg_id].get("decision_grade_reason"),
                }
            )
    return {
        "gate": (
            "CONTRACT.md 'Validity gate' per leg: kernel table present; counter equations hold; "
            "--expect-kernels-disabled satisfied on D legs (profile_421_merge.py verdict); "
            "UNATTRIBUTED <= limits.unattributed_decision_grade_limit of gpu_busy "
            "(profile_421_attribute.py decision_grade)"
        ),
        "legs_total": len(leg_ids),
        "legs_valid": len(leg_ids) - len(failing),
        "legs_failing": failing,
    }


# --------------------------------------------------------------------------- #
# findings — computed sentences, each carrying the raw evidence it was built
# from (contract close-out: HTSAT front-end, CLIP-vision front-end, CLIP
# launch-bound + BF16 busy/wall trade, C-ATTN-HTSAT out-of-tier number)
# --------------------------------------------------------------------------- #
def _per_step(merge_legs: list[dict], leg_id: str) -> dict:
    row = _index_by_leg_id(merge_legs).get(leg_id)
    if row is None or not isinstance(row.get("per_step"), dict):
        raise ArtifactBuildError(f"{leg_id}: no per_step block in --merge-json to build a finding from")
    return row["per_step"]


def _per_step_finite(merge_legs: list[dict], leg_id: str, field: str) -> float:
    return _finite(_per_step(merge_legs, leg_id).get(field), f"{leg_id}: per_step.{field}")


def _per_step_share(merge_legs: list[dict], leg_id: str, field: str) -> float:
    """`_per_step_finite`, domain-checked to `[0, 1]` — for fields that are a
    SHARE of wall/busy time (never a plain duration or count)."""
    return _finite_share(_per_step(merge_legs, leg_id).get(field), f"{leg_id}: per_step.{field}")


def _chain(attribution_legs: list[dict], leg_id: str, chain: str) -> dict:
    row = _index_by_leg_id(attribution_legs).get(leg_id)
    if row is None or chain not in row.get("chains", {}):
        raise ArtifactBuildError(f"{leg_id}: no {chain!r} chain in --attribution-json to build a finding from")
    return row["chains"][chain]


def _chain_share(attribution_legs: list[dict], leg_id: str, chain: str, field: str) -> float:
    """A chain's own share field, domain-checked to `[0, 1]` via
    `_finite_share` — same convention as `_per_step_share` above."""
    return _finite_share(_chain(attribution_legs, leg_id, chain).get(field), f"{leg_id}: chains[{chain!r}].{field}")


def _merge_verdict_problems(merge_by_id: dict[str, dict], leg_ids: tuple[str, ...]) -> list[str]:
    """Empty iff every named leg's own `--merge-json` row has
    `verdict == "VALID"` — a finding built from an INVALID leg's per-step
    numbers would report a number `profile_421_merge.py` itself refused to
    certify (a failed positive-proof equation, a disagreeing checkpoint,
    ...); any finding depending on that leg is suppressed by its caller
    rather than silently trusting the number anyway (contract Method /
    Validity gate; `profile_421_merge.py`'s own `summary.legs_invalid`,
    which `derive_status` already gates the artifact's top-level `status`
    on, is the SAME signal — this is that gate applied per-finding)."""
    problems: list[str] = []
    for leg_id in leg_ids:
        row = merge_by_id.get(leg_id)
        if row is None:
            raise ArtifactBuildError(f"{leg_id}: not present in --merge-json's own legs to gate a finding on")
        verdict = row.get("verdict")
        if verdict != "VALID":
            problems.append(f"{leg_id}: merge verdict is {verdict!r}, not VALID")
    return problems


def _decision_grade_problems(attr_by_id: dict[str, dict], leg_ids: tuple[str, ...]) -> list[str]:
    """Empty iff every named leg's own `--attribution-json` row is
    `decision_grade` — a finding that reads a CHAIN SHARE (as opposed to a
    plain per-step wall/busy number, which `_merge_verdict_problems` above
    already gates) is only as trustworthy as the attribution that produced
    it; a leg whose own attribution is not decision-grade (e.g.
    UNATTRIBUTED over the contract's validity bound) is not a chain share
    this module treats as decided."""
    problems: list[str] = []
    for leg_id in leg_ids:
        row = attr_by_id.get(leg_id)
        if row is None:
            raise ArtifactBuildError(f"{leg_id}: not present in --attribution-json's own legs to gate a finding on")
        if row.get("decision_grade") is not True:
            reason = row.get("decision_grade_reason")
            suffix = f" (recorded reason: {reason})" if isinstance(reason, str) and reason else ""
            problems.append(f"{leg_id}: attribution decision_grade is {row.get('decision_grade')!r}, not True{suffix}")
    return problems


def _contract_validity_problems(
    merge_by_id: dict[str, dict], attr_by_id: dict[str, dict], leg_ids: tuple[str, ...]
) -> list[str]:
    """The contract's ONE validity gate (CONTRACT.md "Validity gate": kernel
    table present; counter equations hold; `--expect-kernels-disabled`
    satisfied on D legs; UNATTRIBUTED <= the recorded bound of gpu_busy),
    applied per leg exactly as pre-registered: the first three clauses are
    `profile_421_merge.py`'s own verdict (`_merge_verdict_problems`), the
    fourth is `profile_421_attribute.py`'s `decision_grade`
    (`_decision_grade_problems`). A leg failing ANY clause is not a valid
    datum for ANY finding — whether the finding reads a chain share or a
    plain per-step wall number. The gate was frozen as one gate; a finding
    that read a per-step number off a leg the fourth clause refuses would be
    splitting that gate after the numbers were seen, and that is exactly
    what a pre-registration exists to forbid."""
    return _merge_verdict_problems(merge_by_id, leg_ids) + _decision_grade_problems(attr_by_id, leg_ids)


def _contract_valid(merge_by_id: dict[str, dict], attr_by_id: dict[str, dict], leg_id: str) -> bool:
    """`True` iff `leg_id` is present in both reports AND passes every
    clause of the contract's validity gate — for OPTIONAL supporting legs
    (the A/D-arm set an invariance clause reads) that a finding may be
    built WITHOUT; a REQUIRED leg goes through `_contract_validity_problems`
    so its absence is a named refusal, never a silent "not applicable"."""
    if leg_id not in merge_by_id or leg_id not in attr_by_id:
        return False
    return not _contract_validity_problems(merge_by_id, attr_by_id, (leg_id,))


def _decision_legs_phrase(merge_by_id: dict[str, dict], legs_read: tuple[str, ...]) -> str:
    """"across the F32/BF16 decision legs" when more than one decision leg
    was read, "on the F32 decision leg (htsat-A1)" when exactly one was —
    the sentence names the legs it was actually built from, never a fixed
    "F32/BF16" that would keep claiming a dtype the gate excluded."""
    dtypes = [str(merge_by_id[leg_id].get("dtype", "")).upper() for leg_id in legs_read]
    if len(legs_read) > 1:
        return f"across the {'/'.join(dtypes)} decision legs"
    return f"on the {dtypes[0]} decision leg ({legs_read[0]})"


def _excluded_legs_clause(legs_excluded: dict[str, str]) -> str:
    if not legs_excluded:
        return ""
    parts = [
        f"{leg_id} is excluded from this finding: it fails the contract's validity gate ({reason})"
        for leg_id, reason in legs_excluded.items()
    ]
    return " " + "; ".join(parts) + "."


def _direction_word(delta_pct: float, *, decrease: str, increase: str) -> str:
    """The sign-derived verb for a BF16-vs-F32 percent delta — `decrease`
    (e.g. `"cuts"`) for a strictly negative delta, `increase` (e.g.
    `"grows"`) for a strictly positive one. `delta_pct` has already been
    routed through `_finite` upstream (never a `NaN`), so the only
    remaining question is the sign; an exact `0.0` is its own, honest
    third case rather than being folded into either verb."""
    if delta_pct < 0.0:
        return decrease
    if delta_pct > 0.0:
        return increase
    return "leaves unchanged"


def _direction_clause(metric: str, delta_pct: float, *, decrease: str, increase: str) -> str:
    """`_direction_word` wrapped into a full clause — the exact-zero arm gets
    its OWN clause shape (`"leaves {metric} unchanged"`) rather than being
    jammed into the `"{verb} {metric} by {pct}%"` template with a `by 0%`
    tail that reads as a (redundant, slightly dishonest-sounding) magnitude
    claim about a delta that is not a magnitude at all."""
    word = _direction_word(delta_pct, decrease=decrease, increase=increase)
    if word == "leaves unchanged":
        return f"leaves {metric} unchanged"
    return f"{word} {metric} by {abs(delta_pct):.0f}%"


def _fmt_range(lo: float, hi: float, *, decimals: int = 0) -> str:
    """Formats a `lo <= hi` pair as `"{lo}-{hi}"` at the given decimal
    precision — UNLESS the two round to the identical string at that
    precision, in which case the degenerate `"32-32"` collapses to the
    single value `"32"` (equal after rounding is not a genuine range, and a
    reader should never see a range whose two ends print identically)."""
    lo_s = f"{lo:.{decimals}f}"
    hi_s = f"{hi:.{decimals}f}"
    if lo_s == hi_s:
        return lo_s
    return f"{lo_s}-{hi_s}"


def invariance_word(axes: list[str]) -> str:
    """The qualitative word an invariance clause may use for the axes the
    legs it read actually span: `["dtype", "arm"]` -> "dtype- and
    arm-invariant", `["arm"]` -> "arm-invariant". Shared with
    `profile_421_render_docs.py` so a rendered restatement can never name
    an axis the finding's own evidence does not carry."""
    if not axes:
        raise ArtifactBuildError("invariance_word: no axes — the clause has nothing to claim invariance over")
    return "- and ".join(axes) + "-invariant"


def _optional_leg_valid(merge_by_id: dict[str, dict], leg_id: str) -> bool:
    """`True` iff `leg_id` is present in `--merge-json`'s own legs AND its
    verdict is `VALID` — for OPTIONAL supporting evidence (the D-arm legs an
    arm-invariance check reads) that a finding may still be built WITHOUT,
    unlike `_merge_verdict_problems`'s REQUIRED legs (whose absence is a
    hard refusal, never silently treated as "not applicable"). A run that
    never measured the D-arm at all (or measured it but it came back
    INVALID) simply does not get the arm-invariance CLAUSE — the finding's
    other, required-leg-gated numbers are entirely unaffected."""
    row = merge_by_id.get(leg_id)
    return isinstance(row, dict) and row.get("verdict") == "VALID"


# Rule thresholds for every QUALITATIVE word `compute_findings` can emit —
# each is recorded (by name and value) in the finding's own `evidence` block
# alongside the numbers it was evaluated against, so a downstream reader
# never has to trust the English alone. When a rule does not hold, the
# finding still builds (from the SAME required legs) but drops the
# qualitative word for a neutral sentence stating the same numbers.
FRONT_END_BOUND_SHARE_OF_WALL_MIN = 0.5  # front_share_of_wall >= this, on EVERY leg read
ARM_INVARIANCE_REL_SPREAD_MAX = 0.05  # (max-min)/mean of front_s_per_step across every A/D leg read
LAUNCH_BOUND_LAUNCHES_PER_STEP_MIN = 1000.0  # launches/step >= this, on EVERY leg read
# residual_s_per_step / wall_s_per_step (the contract's own "launch/sync
# residual = wall - front - busy", CONTRACT.md SS Method) >= this, on EVERY
# leg read — "launch-bound" is licensed by the contract's OWN wall
# decomposition, not an externally-assumed microsecond launch-overhead
# figure this module has no way to independently know.
LAUNCH_BOUND_RESIDUAL_SHARE_OF_WALL_MIN = 0.2


def compute_findings(
    merge_report: dict, attribution_report: dict, legs_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Returns `(findings, suppressed_findings)`. A finding whose required
    leg(s) fail `_merge_verdict_problems`/`_decision_grade_problems` is
    never built — not even to have its arithmetic silently discarded — it
    is instead recorded in `suppressed_findings` as `{"id", "legs",
    "reason"}`, naming exactly which leg(s) and why, per the contract's own
    "the artifact producer gates on [legs_invalid]" (§D5)."""
    merge_legs = merge_report.get("legs", [])
    attribution_legs = attribution_report.get("legs", [])
    merge_by_id = _index_by_leg_id(merge_legs)
    attr_by_id = _index_by_leg_id(attribution_legs)
    findings: list[dict] = []
    suppressed: list[dict] = []

    def _suppress(finding_id: str, leg_ids: tuple[str, ...], problems: list[str]) -> None:
        suppressed.append({"id": finding_id, "legs": list(leg_ids), "reason": "; ".join(problems)})

    # 1) HTSAT front-end share of wall (A1 f32, A2 bf16 decision legs) —
    #    "is CPU front-end-bound" is a NAMED rule (FRONT_END_BOUND_SHARE_OF_
    #    WALL_MIN, evaluated on every REQUIRED leg read), dropped for a
    #    neutral sentence stating the same shares when it does not hold. The
    #    "dtype- and arm-invariant" clause is a SECOND, independent rule
    #    that additionally requires the D-arm legs (OPTIONAL — never a hard
    #    refusal if absent/INVALID) and is phrased about SECONDS, not
    #    shares — arm-invariance is a claim about the front end's absolute
    #    cost not moving, which a share-of-wall ratio cannot show on its own
    #    (wall itself moves across arms as kernels are disabled).
    htsat_legs = ("htsat-A1", "htsat-A2")
    htsat_problems = {
        leg_id: _contract_validity_problems(merge_by_id, attr_by_id, (leg_id,)) for leg_id in htsat_legs
    }
    legs_read = tuple(leg_id for leg_id in htsat_legs if not htsat_problems[leg_id])
    legs_excluded = {leg_id: "; ".join(p) for leg_id, p in htsat_problems.items() if p}
    if not legs_read:
        _suppress("htsat-front-end-bound", htsat_legs, [p for ps in htsat_problems.values() for p in ps])
    else:
        htsat_shares = {
            leg_id: _per_step_share(merge_legs, leg_id, "front_share_of_wall") * 100.0 for leg_id in legs_read
        }
        lo, hi = min(htsat_shares.values()), max(htsat_shares.values())
        front_end_bound = all(share >= FRONT_END_BOUND_SHARE_OF_WALL_MIN * 100.0 for share in htsat_shares.values())
        share_range = _fmt_range(lo, hi)
        legs_phrase = _decision_legs_phrase(merge_by_id, legs_read)
        excluded_clause = _excluded_legs_clause(legs_excluded)
        if front_end_bound:
            bound_clause = (
                f"The HTSAT training step is CPU front-end-bound: front-end share of wall is {share_range}% "
                f"{legs_phrase} (rule: front_share_of_wall >= "
                f"{FRONT_END_BOUND_SHARE_OF_WALL_MIN:.0%} on every leg read; audio decode/resample/STFT/mel "
                f"dominating wall time).{excluded_clause}"
            )
        else:
            # The not-front-end-bound arm names the SAME front-end
            # MECHANISM (audio decode/resample/STFT/mel) but drops
            # "dominating" — a magnitude/comparative word this arm has
            # exactly NOT earned (the share-of-wall rule that would license
            # it did not hold on every leg read); the clause must drop that
            # word rather than keep it regardless of which arm fired.
            bound_clause = (
                f"HTSAT's front-end share of wall is {share_range}% {legs_phrase} (rule: "
                f"front_share_of_wall >= {FRONT_END_BOUND_SHARE_OF_WALL_MIN:.0%} on every leg read was NOT met "
                "on every leg, so 'front-end-bound' is not asserted; the front-end work here is audio "
                f"decode/resample/STFT/mel).{excluded_clause}"
            )

        evidence: dict[str, object] = {
            "front_share_of_wall_pct": htsat_shares,
            "front_end_bound_rule": f"front_share_of_wall >= {FRONT_END_BOUND_SHARE_OF_WALL_MIN} on every leg read",
            "front_end_bound": front_end_bound,
            "legs_read": list(legs_read),
            "legs_excluded": legs_excluded,
        }

        # The invariance clause reads every A/D-arm leg that passes the
        # contract's validity gate. "arm-invariant" needs BOTH arms among
        # the legs read; "dtype-" is added only when BOTH dtypes are — the
        # words name the axes the legs read actually span, never a fixed
        # "dtype- and arm-" that would keep claiming an axis the gate
        # excluded every witness of.
        arm_leg_ids = ("htsat-A1", "htsat-A2", "htsat-D1", "htsat-D2")
        arm_legs_read = tuple(leg_id for leg_id in arm_leg_ids if _contract_valid(merge_by_id, attr_by_id, leg_id))
        arms_read = {str(merge_by_id[leg_id].get("arm")) for leg_id in arm_legs_read}
        dtypes_read = {str(merge_by_id[leg_id].get("dtype")) for leg_id in arm_legs_read}
        invariance_axes = (["dtype"] if {"f32", "bf16"} <= dtypes_read else []) + (["arm"] if {"A", "D"} <= arms_read else [])
        invariance_clause = ""
        if "arm" in invariance_axes:
            front_seconds = {
                leg_id: _per_step_finite(merge_legs, leg_id, "front_s_per_step") for leg_id in arm_legs_read
            }
            smin, smax = min(front_seconds.values()), max(front_seconds.values())
            smean = sum(front_seconds.values()) / len(front_seconds)
            if smax == smin:
                rel_spread = 0.0
            elif smean > 0.0:
                rel_spread = (smax - smin) / smean
            else:
                rel_spread = math.inf
            arm_invariant = rel_spread <= ARM_INVARIANCE_REL_SPREAD_MAX
            seconds_range = _fmt_range(smin, smax, decimals=3)
            axes_word = invariance_word(invariance_axes)
            legs_desc = (
                f"every contract-valid {'/'.join(sorted(d.upper() for d in dtypes_read))} x "
                f"{'/'.join(sorted(arms_read))}-arm leg read ({', '.join(arm_legs_read)})"
            )
            if arm_invariant:
                invariance_clause = (
                    f" Front-end time itself is {seconds_range} s/step across {legs_desc} "
                    f"(relative spread {rel_spread * 100.0:.1f}%, within the "
                    f"{ARM_INVARIANCE_REL_SPREAD_MAX:.0%} arm-invariance rule), so this cost is {axes_word}."
                )
            else:
                invariance_clause = (
                    f" Front-end time is {seconds_range} s/step across {legs_desc} "
                    f"(relative spread {rel_spread * 100.0:.1f}%, above the {ARM_INVARIANCE_REL_SPREAD_MAX:.0%} "
                    f"arm-invariance rule) — not treated as invariant over {'/'.join(invariance_axes)} here."
                )
            arm_excluded = {
                leg_id: "; ".join(_contract_validity_problems(merge_by_id, attr_by_id, (leg_id,)))
                for leg_id in arm_leg_ids
                if leg_id in merge_by_id and leg_id in attr_by_id and not _contract_valid(merge_by_id, attr_by_id, leg_id)
            }
            # A D-arm leg the gate excluded is named HERE (the decision legs'
            # own exclusions are already named in the primary clause).
            clause_only = {leg_id: r for leg_id, r in arm_excluded.items() if leg_id not in legs_excluded}
            if clause_only:
                invariance_clause += " " + "; ".join(
                    f"{leg_id} is excluded from this clause: it fails the contract's validity gate ({reason})"
                    for leg_id, reason in clause_only.items()
                ) + "."
            evidence["front_s_per_step"] = front_seconds
            evidence["arm_invariance_rule"] = (
                f"(max-min)/mean of front_s_per_step across every contract-valid A/D leg read <= "
                f"{ARM_INVARIANCE_REL_SPREAD_MAX}"
            )
            evidence["arm_invariance_relative_spread"] = rel_spread
            evidence["arm_invariant"] = arm_invariant
            evidence["invariance_axes"] = invariance_axes
            evidence["arm_invariance_legs_read"] = list(arm_legs_read)
            evidence["arm_invariance_legs_excluded"] = arm_excluded

        findings.append({"id": "htsat-front-end-bound", "text": bound_clause + invariance_clause, "evidence": evidence})

    # 2) CLIP-vision front-end share of wall (A1 f32, A2 bf16 decision legs).
    vision_legs = ("clip-vision-A1", "clip-vision-A2")
    problems = _contract_validity_problems(merge_by_id, attr_by_id, vision_legs)
    if problems:
        _suppress("clip-vision-front-end-share", vision_legs, problems)
    else:
        vision_shares = {
            leg_id: _per_step_share(merge_legs, leg_id, "front_share_of_wall") * 100.0 for leg_id in vision_legs
        }
        lo, hi = min(vision_shares.values()), max(vision_shares.values())
        findings.append(
            {
                "id": "clip-vision-front-end-share",
                "text": (
                    f"CLIP-vision's image decode/preprocess front end is {_fmt_range(lo, hi)}% of wall on the "
                    "F32/BF16 decision legs."
                ),
                "evidence": {"front_share_of_wall_pct": vision_shares},
            }
        )

    # 3) CLIP launch-bound at batch 8, + the BF16 busy/wall trade.
    launch_legs = ("clip-text-A1", "clip-text-A2", "clip-vision-A1", "clip-vision-A2")
    problems = _contract_validity_problems(merge_by_id, attr_by_id, launch_legs)
    if problems:
        _suppress("clip-launch-bound-batch8", launch_legs, problems)
    else:
        legs_by_id = _index_by_leg_id([{"leg_id": p.name} for p in _leg_dirs(legs_dir)])
        launches: dict[str, float] = {}
        for leg_id in launch_legs:
            if leg_id not in legs_by_id:
                raise ArtifactBuildError(f"{leg_id}: not present under --legs-dir; cannot build the launch-bound finding")
            census = _leg_file(legs_dir, leg_id, "census.json", f"{leg_id}: census.json")
            launches[leg_id] = _finite(census.get("launches_per_step"), f"{leg_id}: census.json launches_per_step")
        busy_deltas_pct: dict[str, float] = {}
        wall_deltas_pct: dict[str, float] = {}
        tower_order = ("clip-text", "clip-vision")
        for tower, a1, a2 in (
            ("clip-text", "clip-text-A1", "clip-text-A2"),
            ("clip-vision", "clip-vision-A1", "clip-vision-A2"),
        ):
            busy_a1 = _per_step_finite(merge_legs, a1, "busy_s_per_step")
            busy_a2 = _per_step_finite(merge_legs, a2, "busy_s_per_step")
            wall_a1 = _per_step_finite(merge_legs, a1, "wall_s_per_step")
            wall_a2 = _per_step_finite(merge_legs, a2, "wall_s_per_step")
            # `busy_a1`/`wall_a1` are this delta's own denominator — a zero
            # denominator is exactly as uncomputable as a NaN numerator, not
            # a divide-by-zero this module should let Python turn into an
            # `inf` or a `ZeroDivisionError` several lines away from its
            # real cause.
            if busy_a1 == 0.0:
                raise ArtifactBuildError(f"{a1}: per_step.busy_s_per_step is 0 — cannot compute a BF16-vs-F32 busy delta")
            if wall_a1 == 0.0:
                raise ArtifactBuildError(f"{a1}: per_step.wall_s_per_step is 0 — cannot compute a BF16-vs-F32 wall delta")
            busy_deltas_pct[tower] = (busy_a2 - busy_a1) / busy_a1 * 100.0
            wall_deltas_pct[tower] = (wall_a2 - wall_a1) / wall_a1 * 100.0

        # "are launch-bound" is a NAMED rule too: launches/step above a
        # floor AND the contract's own launch/sync RESIDUAL a material share
        # of wall, both true on every named leg — never a hard-coded verb
        # regardless of what the numbers say.
        residual_shares: dict[str, float] = {}
        for leg_id in launch_legs:
            residual_s = _per_step_finite(merge_legs, leg_id, "residual_s_per_step")
            wall_s = _per_step_finite(merge_legs, leg_id, "wall_s_per_step")
            if wall_s == 0.0:
                raise ArtifactBuildError(f"{leg_id}: per_step.wall_s_per_step is 0 — cannot compute a residual share of wall")
            residual_shares[leg_id] = residual_s / wall_s
        launch_bound = all(
            launches[leg_id] >= LAUNCH_BOUND_LAUNCHES_PER_STEP_MIN
            and residual_shares[leg_id] >= LAUNCH_BOUND_RESIDUAL_SHARE_OF_WALL_MIN
            for leg_id in launch_legs
        )
        launch_lo, launch_hi = min(launches.values()), max(launches.values())
        launch_range = _fmt_range(launch_lo, launch_hi)
        launch_bound_rule = (
            f"launches/step >= {LAUNCH_BOUND_LAUNCHES_PER_STEP_MIN:.0f} and launch/sync residual share of wall "
            f"(residual_s_per_step / wall_s_per_step) >= {LAUNCH_BOUND_RESIDUAL_SHARE_OF_WALL_MIN:.0%}, both on "
            "every named leg"
        )
        if launch_bound:
            launch_sentence = (
                f"At batch 8 the CLIP training steps are launch-bound: {launch_range} launches/step across the "
                f"four F32/BF16 A-arm CLIP legs (text and vision) (rule: {launch_bound_rule}); "
            )
        else:
            launch_sentence = (
                f"At batch 8 the CLIP training steps issue {launch_range} launches/step across the four "
                f"F32/BF16 A-arm CLIP legs (text and vision) (rule: {launch_bound_rule} was NOT true on every "
                "leg, so 'launch-bound' is not asserted); "
            )
        evidence: dict[str, object] = {
            "launches_per_step": launches,
            "busy_delta_pct_bf16_vs_f32": busy_deltas_pct,
            "wall_delta_pct_bf16_vs_f32": wall_deltas_pct,
            "residual_share_of_wall": residual_shares,
            "launch_bound_rule": launch_bound_rule,
            "launch_bound": launch_bound,
        }
        if all(v < 0.0 for v in busy_deltas_pct.values()) and all(v < 0.0 for v in wall_deltas_pct.values()):
            # Every tower's delta is negative (BF16 cheaper everywhere): a
            # single blanket "cuts"/"drops" verb is honest here, so the
            # sentence reports the MAGNITUDE range (`abs(...)`, ordered by
            # ascending magnitude via `key=abs` rather than a plain
            # `min`/`max` pick) rather than repeating the sign per tower —
            # a signed "-32...-41%" would double-negate (a cut that is
            # itself negative reads as a GROWTH, the opposite of what the
            # number means). `_fmt_range` collapses a degenerate equal-after-
            # -rounding pair (e.g. "1-1%") to the single value "1%".
            busy_by_magnitude = sorted(busy_deltas_pct.values(), key=abs)
            wall_by_magnitude = sorted(wall_deltas_pct.values(), key=abs)
            busy_range = _fmt_range(abs(busy_by_magnitude[0]), abs(busy_by_magnitude[-1]))
            wall_range = _fmt_range(abs(wall_by_magnitude[0]), abs(wall_by_magnitude[-1]))
            # "only" is itself a NAMED rule: it claims wall moved LESS than
            # busy did, so it is licensed iff that magnitude comparison
            # holds POINTWISE on every tower — never assumed just because
            # both deltas happen to be negative.
            wall_smaller_every_tower = all(
                abs(wall_deltas_pct[tower]) < abs(busy_deltas_pct[tower]) for tower in tower_order
            )
            only_word = "only " if wall_smaller_every_tower else ""
            text = (
                launch_sentence
                + "switching to "
                f"BF16 cuts GPU busy by {busy_range}% per tower while wall drops by {only_word}{wall_range}%."
            )
            evidence["wall_drop_smaller_than_busy_drop_every_tower"] = wall_smaller_every_tower
        else:
            # A mixed-sign pair across towers, or a non-negative delta: a
            # blanket "cuts" would be dishonest for whichever tower actually
            # GREW, so each tower gets its own sign-derived clause instead
            # of a single magnitude range (`_direction_clause` gives an
            # exact-zero delta its own "leaves X unchanged" shape rather
            # than a degenerate "by 0%").
            per_tower_sentences = []
            for tower in tower_order:
                busy_clause = _direction_clause("GPU busy", busy_deltas_pct[tower], decrease="cuts", increase="grows")
                wall_clause = _direction_clause("wall", wall_deltas_pct[tower], decrease="cuts", increase="grows")
                per_tower_sentences.append(f"{tower} BF16 {busy_clause} and {wall_clause} vs F32")
            text = (
                launch_sentence
                + "BF16-vs-F32 per tower (not uniformly one direction, so no single verb applies): "
                + "; ".join(per_tower_sentences)
                + "."
            )
        findings.append({"id": "clip-launch-bound-batch8", "text": text, "evidence": evidence})

    # 4) C-ATTN-HTSAT: measured, out-of-tier (declared out of scope for a
    #    port decision under this contract — a NUMBER, never a verdict).
    #    Reads a CHAIN SHARE, so this gates on decision_grade too, not just
    #    the merge verdict.
    c_attn_legs = ("htsat-A1",)
    problems = _contract_validity_problems(merge_by_id, attr_by_id, c_attn_legs)
    if problems:
        _suppress("c-attn-htsat-out-of-tier", c_attn_legs, problems)
    else:
        share_gpu_busy = _chain_share(attribution_legs, "htsat-A1", "C-ATTN-htsat", "share_gpu_busy")
        share_wall = _chain_share(attribution_legs, "htsat-A1", "C-ATTN-htsat", "share_wall")
        busy_pct = share_gpu_busy * 100.0
        wall_pct = share_wall * 100.0
        findings.append(
            {
                "id": "c-attn-htsat-out-of-tier",
                "text": (
                    f"C-ATTN-HTSAT is measured, not a candidate port: {busy_pct:.0f}% of GPU busy "
                    f"(~{wall_pct:.0f}% of wall) on the F32 decision leg (htsat-A1). HTSAT attention "
                    "(head_dim 24 at every stage) sits OUTSIDE the fixed-head-dim port tier by the "
                    "contract's own declaration — this stays a measured, OPEN number, never folded into "
                    "UNATTRIBUTED and never decided under this issue."
                ),
                "evidence": {"leg_id": "htsat-A1", "share_gpu_busy": share_gpu_busy, "share_wall": share_wall},
            }
        )

    return findings, suppressed


def derive_status(merge_report: dict) -> str:
    """GREEN iff `--merge-json`'s own `summary` reports zero invalid legs
    AND zero failed P2 towers — read live off that summary, never hand-
    declared by the identity sidecar. A sidecar that could simply assert
    `"status": "GREEN"` is exactly the "restored coverage" transcription
    failure mode the house principles name: nothing would recompute it if
    the underlying run actually went RED."""
    summary = merge_report.get("summary")
    if not isinstance(summary, dict):
        raise ArtifactBuildError("--merge-json has no summary block to derive status from")
    legs_invalid = summary.get("legs_invalid")
    p2_fail = summary.get("p2_fail")
    if not isinstance(legs_invalid, int) or isinstance(legs_invalid, bool):
        raise ArtifactBuildError(f"--merge-json summary.legs_invalid is not an int ({legs_invalid!r})")
    if not isinstance(p2_fail, int) or isinstance(p2_fail, bool):
        raise ArtifactBuildError(f"--merge-json summary.p2_fail is not an int ({p2_fail!r})")
    if legs_invalid == 0 and p2_fail == 0:
        return "GREEN"
    return f"RED: {legs_invalid} leg(s) INVALID, {p2_fail} P2 tower(s) FAIL"


def compute_kernel_identity_split_count(legs_dir: Path, leg_id: str, coalesced_name: str) -> int:
    """The number of DISTINCT post-fix kernel names that a single pre-fix
    `by_kernel_and_grid` bucket named `coalesced_name` (e.g. `"Kernel2"`,
    cutlass's template-wrapper shortName) was resplit into by
    `kernel_census.py`'s `COALESCE(demangledName, shortName)` keying — read
    live off `<leg_id>/census.pre-demangle.json` (the pre-fix, shortName-
    only buckets) cross-referenced against `<leg_id>/census.json` (the
    post-fix, demangled buckets) at the SAME `(grid, block)`, never a
    hand-typed count. The identity sidecar's own "N DISTINCT ... instantiations
    sharing shortName=... AND grid/block" prose quotes the MAXIMUM split any
    single `coalesced_name` bucket underwent for this leg."""
    pre = _leg_file(legs_dir, leg_id, "census.pre-demangle.json", f"{leg_id}: census.pre-demangle.json")
    post = _leg_file(legs_dir, leg_id, "census.json", f"{leg_id}: census.json")
    pre_rows = pre.get("by_kernel_and_grid")
    post_rows = post.get("by_kernel_and_grid")
    if not isinstance(pre_rows, list) or not isinstance(post_rows, list):
        raise ArtifactBuildError(
            f"{leg_id}: by_kernel_and_grid missing (or not a list) in census.json or census.pre-demangle.json"
        )
    post_names_by_grid_block: dict[tuple, set[str]] = {}
    for row in post_rows:
        key = (tuple(row["grid"]), tuple(row["block"]))
        post_names_by_grid_block.setdefault(key, set()).add(row["kernel"])
    best = 0
    for row in pre_rows:
        if row.get("kernel") != coalesced_name:
            continue
        key = (tuple(row["grid"]), tuple(row["block"]))
        best = max(best, len(post_names_by_grid_block.get(key, set())))
    if best == 0:
        raise ArtifactBuildError(
            f"{leg_id}: no {coalesced_name!r} bucket found in census.pre-demangle.json to compute a split count from"
        )
    return best


# --------------------------------------------------------------------------- #
# identity-sidecar deviation TEMPLATES — every number the prose quotes is
# filled in here from a live source, never hand-typed into the sidecar
# itself (see `profile_421_run2_identity.json`'s own header comment).
# --------------------------------------------------------------------------- #
# The EXACT string `profile_421_attribute.py`'s own `leg_decision_grade`
# writes into a leg's `decision_grade_reason` when (and ONLY when) that
# leg's UNATTRIBUTED share is over the validity bound — mirrored here
# (never imported: it is a STRING SHAPE this run's own recorded record
# produced, not a regex `profile_421_attribute.py` itself exposes) so this
# producer can detect that SHAPE and cross-check the reason's own SECOND,
# independent copy of `share` against `chains.UNATTRIBUTED.share_gpu_busy`
# (`_htsat_bound_deviation_context` below) — the BOUND itself is never read
# off this string (see `_recorded_limit`); a leg that failed
# `leg_decision_grade` for an unrelated reason records no such string to
# parse a bound out of in the first place. Also mirrored (independently) by
# `profile_421_render_docs.py`'s own `_DECISION_GRADE_REASON_RE` — the two
# must parse the SAME shape, and each refuses closed on a mismatch rather
# than silently diverge.
_UNATTRIBUTED_OVER_BOUND_REASON_RE = re.compile(r"^UNATTRIBUTED share_gpu_busy=([0-9.eE+-]+) > ([0-9.eE+-]+)$")


def _recorded_limit(attribution_report: dict, key: str, live_value: float) -> float:
    """The validity-gate bound named `key` THIS RUN's own `--attribution-
    json` actually recorded, in its own top-level `limits` block
    (`profile_421_attribute.py`'s `build_report` stamps this at
    report-build time from its OWN live constants — module doc, "`limits`:
    the report's own recorded validity-gate bounds") — the run's own
    written record of what it enforced, never re-parsed out of any leg's
    own `decision_grade_reason` string (a leg that failed for an unrelated
    reason, or never failed at all, records no such string) and never the
    live `_attribute_mod` import directly.

    Refuses BY NAME if `limits` (or `key` within it) is absent — an
    `--attribution-json` built before this block existed, or missing a
    bound some future decision reads, carries no record for this producer
    to trust; guessing (falling back to the live constant) would silently
    re-judge an already-measured run against a bound it may never have
    actually been held to at measurement time.

    `live_value` (the CURRENT `profile_421_attribute.py` constant of the
    SAME name) is consulted ONLY to refuse the build if it disagrees with
    the recorded value — naming BOTH — never to supply a bound this run's
    own record lacks: `profile_421_attribute.py`'s own constant can move (a
    later, unrelated PR edits it) without this ALREADY-MEASURED run's own
    artifact silently re-judging itself against a bound it was never
    actually held to."""
    limits = attribution_report.get("limits")
    if not isinstance(limits, dict):
        raise ArtifactBuildError(
            "--attribution-json carries no top-level 'limits' block — regenerate it with a "
            f"profile_421_attribute.py new enough to record its own validity-gate bounds (needed here for {key!r})"
        )
    if key not in limits:
        raise ArtifactBuildError(
            f"--attribution-json's own 'limits' block carries no {key!r} entry — regenerate it with a "
            "profile_421_attribute.py new enough to record this bound"
        )
    recorded = limits[key]
    if not isinstance(recorded, (int, float)) or isinstance(recorded, bool):
        raise ArtifactBuildError(f"--attribution-json limits[{key!r}] is not a number ({recorded!r})")
    if recorded != live_value:
        raise ArtifactBuildError(
            f"profile_421_attribute.py's own {key} ({live_value!r}) disagrees with the value this run's "
            f"own --attribution-json recorded in limits[{key!r}] ({recorded!r}) — the constant has moved "
            "since this run was measured; a regeneration must not silently re-judge an already-measured "
            "run against a different bound than the one it was actually held to at measurement time"
        )
    return float(recorded)


def _resolve_recorded_limits(attribution_report: dict) -> dict[str, float]:
    """Both validity-gate bounds this artifact's `limits` block and its
    identity-sidecar prose need, resolved ONCE (never re-read per call site,
    so every consumer — the sidecar's htsat clauses, the top-level
    `report["limits"]`, the "N% known-kernel-name gate" prose — agrees on
    the exact same recorded value). See `_recorded_limit`'s own doc."""
    return {
        "unattributed_decision_grade_limit": _recorded_limit(
            attribution_report, "unattributed_decision_grade_limit", _attribute_mod.UNATTRIBUTED_DECISION_GRADE_LIMIT
        ),
        "unknown_kernel_share_limit": _recorded_limit(
            attribution_report, "unknown_kernel_share_limit", _attribute_mod.UNKNOWN_KERNEL_SHARE_LIMIT
        ),
    }


def _format_recorded_bound_pct(value: float) -> str:
    """A recorded validity-gate bound, already multiplied by 100, printed
    EXACTLY as recorded — `5.0 -> "5%"`, `5.5 -> "5.5%"` — never a fixed
    `.0f`/`.2f` spec that would silently round a non-round bound down (or
    dress a round one up in false precision). `%.6f` then a trailing-
    zero/point strip absorbs ordinary float noise (e.g. `0.055 * 100.0`)
    without inventing digits a bound was never measured to."""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return f"{text}%"


def _htsat_bound_deviation_context(
    prefix: str,
    leg_id: str,
    merge_by_id: dict[str, dict],
    attr_by_id: dict[str, dict],
    attribution_legs: list[dict],
    bound: float,
) -> dict[str, object]:
    """Every placeholder ONE htsat leg's own deviation clause needs, read
    live off THAT SAME leg's own merge/attribution rows, keyed under the
    given `prefix` (`"htsat_a1"` / `"htsat_a2"`) so the SAME mechanism
    serves every leg the identity sidecar's prose names — never hand-typed,
    and never one flag answering two different questions. `bound` is the
    run's own RECORDED validity bound (`_resolve_recorded_limits`, resolved
    once for the whole report off `attribution_report["limits"]`) — this
    function never reads the live import itself, so a constant that moved
    since the run was measured cannot silently change what this ONE leg's
    own clause quotes independently of the sibling leg.

    The sentence this feeds ALWAYS states the leg's own recorded
    `decision_grade_reason` VERBATIM (`{prefix}_recorded_reason_clause`,
    non-empty whenever `decision_grade` is `False`) and reports the
    share/bound relation (`{prefix}_bound_comparison_word`) as a SEPARATE
    clause, asserted as a fact this function itself checked — never as the
    reason's claimed CAUSE, unless the reason string literally IS the
    share-bound reason (in which case the two clauses simply agree, rather
    than one silently standing in for the other on a run where they
    diverge). `decision_grade=True` forces `reason` to `None`
    (`leg_decision_grade` never returns `True` with one), so
    `{prefix}_recorded_reason_clause` is always empty in that arm — there
    is nothing recorded to quote.

    `leg_decision_grade` (`profile_421_attribute.py`) has EIGHT return
    points: one `True`, and seven distinct `False` reasons, of which "the
    UNATTRIBUTED share is over the bound" is only ONE. This function
    therefore handles the full `(decision_grade, over_bound, reason_is_
    share_bound_shape)` lattice explicitly, where `over_bound` is
    `share > bound` computed against the SAME recorded `bound` this run was
    actually judged against, never re-derived from `decision_grade` itself:

    - `(True, under, *)`: decision-grade, genuinely at-or-under the bound —
      the ordinary case; no reason was ever recorded to quote.
    - `(True, over, *)`: impossible from `profile_421_attribute.py`'s own
      `leg_decision_grade` (which never returns `True` without itself
      having confirmed `share <= bound` against this exact bound) — refused
      by name, never rendered, since a live artifact reporting this
      combination is this producer's own inputs contradicting each other.
    - `(False, over, share-shape)`: not decision-grade, and genuinely over
      the bound, for exactly the reason recorded — the leg's own
      `decision_grade_reason` is parsed for a SECOND, independent copy of
      `share` and cross-checked to agree with `chains.UNATTRIBUTED.
      share_gpu_busy` before being trusted; a disagreement refuses rather
      than picks one.
    - `(False, over, not-share-shape)`: not decision-grade for an UNRELATED
      reason (stated verbatim) while the raw share INDEPENDENTLY also
      happens to be over `bound` — both facts are true and both are
      stated, but the sentence never claims the recorded reason IS this
      bound (it demonstrably is not: its own text does not match the
      share-bound shape at all).
    - `(False, under, not-share-shape)`: not decision-grade for an
      unrelated reason (stated verbatim) while the share genuinely clears
      the bound.
    - `(False, under, share-shape)`: impossible — the reason CLAIMS an
      over-bound failure while this function's own comparison against the
      SAME bound says the share is at-or-under — a self-contradiction in
      the run's own record, refused rather than rendered as either
      direction.

    A leg that is not merge-VALID, or carries no UNATTRIBUTED chain at all,
    is refused HERE, never silently rendered with a stale claim: a
    deviation quoting `{prefix}_bound_comparison_word`/`{prefix}_bound_
    clears_word` for a SUPPRESSED leg would be exactly the kind of
    self-contradiction a reader could never detect from the prose alone."""
    merge_row = merge_by_id.get(leg_id)
    if merge_row is None:
        raise ArtifactBuildError(
            f"{leg_id}: not present in --merge-json's own legs to build the identity template context from"
        )
    merge_verdict = merge_row.get("verdict")
    if not isinstance(merge_verdict, str) or not merge_verdict:
        raise ArtifactBuildError(f"{leg_id}: --merge-json verdict is not a non-empty string ({merge_verdict!r})")
    if merge_verdict != "VALID":
        raise ArtifactBuildError(
            f"{leg_id}: --merge-json verdict is {merge_verdict!r}, not VALID — the identity sidecar's own "
            f"{leg_id} deviation assumes this leg IS merge-VALID; update that deviation (and this gate) "
            "before quoting its chain share under a different state"
        )

    attr_row = attr_by_id.get(leg_id)
    if attr_row is None:
        raise ArtifactBuildError(
            f"{leg_id}: not present in --attribution-json's own legs to build the identity template context from"
        )
    decision_grade = attr_row.get("decision_grade")
    if not isinstance(decision_grade, bool):
        raise ArtifactBuildError(f"{leg_id}: --attribution-json decision_grade is not a bool ({decision_grade!r})")

    share = _chain_share(attribution_legs, leg_id, _attribute_mod.CHAIN_UNATTRIBUTED, "share_gpu_busy")
    share_pct = share * 100.0
    reason = attr_row.get("decision_grade_reason")
    match = _UNATTRIBUTED_OVER_BOUND_REASON_RE.match(reason) if isinstance(reason, str) else None
    reason_is_share_bound = match is not None

    if match is not None:
        # This leg's OWN recorded reason carries a SECOND, independent copy
        # of `share` — cross-checked against the artifact's own
        # `chains.UNATTRIBUTED.share_gpu_busy` before being trusted; a
        # disagreement refuses rather than silently picks one. The
        # reason's OWN bound (`match.group(2)`) is never read for anything
        # — `bound` comes from `attribution_report["limits"]`
        # (`_resolve_recorded_limits`), independent of this string entirely.
        reason_share = float(match.group(1))
        if round(reason_share, 4) != round(share, 4):
            raise ArtifactBuildError(
                f"{leg_id}: decision_grade_reason share ({reason_share!r}) disagrees with the same "
                f"leg's own chains['UNATTRIBUTED']['share_gpu_busy'] ({share!r}) — these two artifact "
                "fields must agree; refuse rather than silently pick one"
            )

    # `over_bound` is computed against the SAME recorded `bound` this run
    # was actually judged against — never re-derived from `decision_grade`
    # itself (`decision_grade` is `False` for six OTHER reasons that have
    # nothing to do with this bound at all).
    over_bound = share > bound

    if decision_grade and over_bound:
        # (True, over): impossible from `leg_decision_grade`'s own
        # mechanism — a live artifact reporting both is this producer's
        # own inputs contradicting each other, never a case this module's
        # prose may paper over.
        raise ArtifactBuildError(
            f"{leg_id}: decision_grade is True but chains['UNATTRIBUTED']['share_gpu_busy'] ({share!r}) "
            f"is over the run's own resolved bound ({bound!r}) — this is impossible from "
            "profile_421_attribute.py's own leg_decision_grade and is refused rather than rendered"
        )
    if (not decision_grade) and (not over_bound) and reason_is_share_bound:
        # (False, under) but the leg's own reason IS the share-bound shape:
        # the reason claims an over-bound failure while this function's own
        # comparison (against the SAME bound the reason itself named) says
        # the share is at-or-under — the numeric cross-checks above already
        # guard the two NUMBERS agreeing; this guards the DIRECTION too.
        raise ArtifactBuildError(
            f"{leg_id}: decision_grade_reason ({reason!r}) reports an over-bound failure but the share "
            f"({share!r}) is at-or-under the resolved bound ({bound!r}) by this module's own comparison "
            "— refuse rather than render a self-contradictory deviation"
        )

    comparison_word = "over" if over_bound else "at or under"
    # ALWAYS state the leg's own recorded reason VERBATIM whenever
    # `decision_grade` is `False` — regardless of `over_bound`/`reason_is_
    # share_bound` — never only for the "unrelated reason" cell: a reader
    # must see the ACTUAL recorded reason, and the share/bound comparison
    # this function computed is a SEPARATE, independently-true clause
    # (`{prefix}_bound_comparison_word`) the caller's own template states
    # next to it, never folded into this clause as a claimed cause. Empty
    # when `decision_grade` is `True` — `leg_decision_grade` never records a
    # reason in that arm, so there is nothing to quote.
    if decision_grade:
        recorded_reason_clause = ""
    else:
        display_reason = reason if isinstance(reason, str) and reason else None
        if display_reason is None:
            attribution_verdict = attr_row.get("verdict")
            display_reason = (
                f"attribution verdict is {attribution_verdict!r}, not VALID"
                if attribution_verdict != _attribute_mod.VERDICT_VALID
                else "reason not recorded"
            )
        recorded_reason_clause = f" (recorded reason: {display_reason})"

    return {
        f"{prefix}_merge_verdict": merge_verdict,
        f"{prefix}_decision_grade_word": "decision-grade" if decision_grade else "NOT decision-grade",
        # The status word for a deviation's own closing "so {leg}'s own
        # {word} status never blocks a candidate-port decision" clause —
        # that claim (HTSAT has no candidate port under this contract in
        # the first place) is true regardless of this leg's own decision-
        # grade state, so the clause itself is licensed in BOTH arms; only
        # the STATUS WORD it names must flip with the SAME live flag.
        f"{prefix}_decision_grade_status_word": "decision-grade" if decision_grade else "non-decision-grade",
        f"{prefix}_bound_comparison_word": comparison_word,
        # A second phrasing of the SAME `over_bound` fact ("clears"/"does
        # not clear the bound") for a deviation whose own prose reads as a
        # verb rather than an "over/at or under" comparison (htsat-A1's own
        # "clears the bound ... and IS decision-grade" clause).
        f"{prefix}_bound_clears_word": "does not clear" if over_bound else "clears",
        f"{prefix}_decision_grade_is_word": "IS" if decision_grade else "is NOT",
        f"{prefix}_unattributed_share_gpu_busy_pct": share_pct,
        f"{prefix}_recorded_reason_clause": recorded_reason_clause,
    }


def _identity_template_context(
    merge_report: dict, attribution_report: dict, legs_dir: Path, recorded_limits: dict[str, float]
) -> dict[str, object]:
    attribution_legs = attribution_report.get("legs", [])
    merge_by_id = _index_by_leg_id(merge_report.get("legs", []))
    attr_by_id = _index_by_leg_id(attribution_legs)
    pool = compute_media_corpus_pool()

    # `recorded_limits` is resolved ONCE, by the caller (`build_report`), off
    # this SAME run's own `attribution_report["limits"]` (`_resolve_recorded_
    # limits`) — never re-read per leg here, so a constant that moved
    # between two calls could never make the two htsat clauses (or the
    # top-level `report["limits"]` this same dict also feeds) disagree on
    # what bound this run was actually judged against.
    bound = recorded_limits["unattributed_decision_grade_limit"]

    context: dict[str, object] = {}
    # Both htsat decision legs the identity sidecar's own deviation prose
    # quotes — htsat-A1 ("clears the bound ... and IS decision-grade") and
    # htsat-A2 ("is {verdict} ... but {word} for attribution") — through
    # the SAME mechanism, so neither can carry a stale hand-typed claim the
    # other's own gate would have refused.
    context.update(_htsat_bound_deviation_context("htsat_a1", "htsat-A1", merge_by_id, attr_by_id, attribution_legs, bound))
    context.update(_htsat_bound_deviation_context("htsat_a2", "htsat-A2", merge_by_id, attr_by_id, attribution_legs, bound))
    # What failing the gate COSTS: the leg is excluded from every finding
    # that would read it — stated next to the failure, never left for a
    # reader to infer from a headline range that quietly narrowed.
    htsat_decision_read = [
        leg_id for leg_id in ("htsat-A1", "htsat-A2") if _contract_valid(merge_by_id, attr_by_id, leg_id)
    ]
    if _contract_valid(merge_by_id, attr_by_id, "htsat-A2"):
        consequence = ""
    elif htsat_decision_read:
        consequence = (
            " It is therefore excluded from every finding that would read it: the HTSAT front-end finding "
            f"reads {', '.join(htsat_decision_read)} only and asserts no dtype-invariance."
        )
    else:
        consequence = (
            " It is therefore excluded from every finding that would read it, and with no contract-valid "
            "HTSAT decision leg left the HTSAT front-end finding is suppressed."
        )
    context["htsat_a2_findings_consequence_clause"] = consequence
    context.update(
        {
            # The RUN'S OWN recorded bounds (`recorded_limits` — see
            # `_resolve_recorded_limits`) — never the live import directly —
            # so the sidecar's own printed "N% bound"/"N% gate" always
            # agrees with the two htsat clauses (and the top-level
            # `report["limits"]`) it sits next to, even if
            # `_attribute_mod`'s own constants have since moved. Formatted
            # EXACTLY as recorded (`_format_recorded_bound_pct`) — never a
            # fixed `.0f`/`.2f` that could round a non-round bound.
            "unattributed_decision_grade_limit_pct": _format_recorded_bound_pct(bound * 100.0),
            "unknown_kernel_share_limit_pct": _format_recorded_bound_pct(
                recorded_limits["unknown_kernel_share_limit"] * 100.0
            ),
            "sqlite_raw_export_count": compute_sqlite_raw_export_count(legs_dir),
            "corpus_total_files": pool["corpus_total_files"],
            "corpus_train_clips": pool["corpus_train_clips"],
            "corpus_rows_m": pool["corpus_rows_m"],
            "kernel_identity_split_count": compute_kernel_identity_split_count(
                legs_dir, KERNEL_IDENTITY_SPLIT_LEGS[0], "Kernel2"
            ),
            "leg_count": len(_leg_dirs(legs_dir)),
            "merge_suite_test_count": compute_merge_suite_test_count(),
        }
    )
    return context


def _render_recorded_deviations(templates: object, context: dict[str, object]) -> list[str]:
    """`--identity`'s own `recorded_deviations` are TEMPLATES (plain prose,
    optionally carrying `{name}`/`{name:.2f}`-style placeholders this
    module fills from `context`) — never a pre-baked string carrying a
    number nothing here recomputed. A template naming a placeholder not in
    `context` is a refusal (`ArtifactBuildError`), never a silently-emitted
    literal `{typo}` in the artifact."""
    if not isinstance(templates, list):
        raise ArtifactBuildError(f"--identity recorded_deviations must be a list, got {templates!r}")
    rendered: list[str] = []
    for i, template in enumerate(templates):
        if not isinstance(template, str):
            raise ArtifactBuildError(f"--identity recorded_deviations[{i}] is not a string ({template!r})")
        try:
            rendered.append(template.format(**context))
        except (KeyError, IndexError, ValueError) as exc:
            raise ArtifactBuildError(
                f"--identity recorded_deviations[{i}] references an unfillable template placeholder: {exc}"
            ) from exc
    return rendered


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #
def build_report(
    legs_dir: Path,
    p2_dir: Path | None,
    merge_report: dict,
    attribution_report: dict,
    identity: dict,
) -> dict:
    p2_towers = p2_tower_names(merge_report)
    git_sha, box, witnessed_p2_towers = collect_identity(legs_dir, p2_dir, p2_towers)
    legs = build_legs(merge_report, attribution_report, legs_dir)
    merge_legs = merge_report.get("legs", [])
    checkpoint_sha256 = collect_checkpoint_sha256(merge_legs)

    checkpoints_notes: dict[str, dict] = {}
    for family, block in identity.get("checkpoints", {}).items():
        entry = dict(block)
        if family in checkpoint_sha256:
            entry["sha256"] = checkpoint_sha256[family]
        checkpoints_notes[family] = entry

    findings, suppressed_findings = compute_findings(merge_report, attribution_report, legs_dir)
    status = derive_status(merge_report)
    contract_validity = build_contract_validity(merge_report, attribution_report, legs_dir)
    # Resolved ONCE, off THIS run's own `attribution_report["limits"]"
    # (`_resolve_recorded_limits`), and threaded to both the sidecar's own
    # template context and the top-level `report["limits"]` below — a
    # single source of truth for every bound a downstream reader
    # (`profile_421_render_docs.py`'s `render_htsat_a2_deviation`, this
    # module's own identity-sidecar clauses) needs.
    recorded_limits = _resolve_recorded_limits(attribution_report)
    template_context = _identity_template_context(merge_report, attribution_report, legs_dir, recorded_limits)
    recorded_deviations = _render_recorded_deviations(identity.get("recorded_deviations", []), template_context)
    p2_witnessed = {"towers": witnessed_p2_towers, "git_sha": git_sha, "box": box} if witnessed_p2_towers else None

    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha,
        "box": box,
        "p2_witnessed": p2_witnessed,
        # This run's own recorded validity-gate bounds (`attribution_report
        # ["limits"]`, cross-checked to agree with the live
        # `profile_421_attribute.py` constants — `_resolve_recorded_
        # limits`) — the ONE place a downstream reader of the FINAL
        # artifact (never `--attribution-json` directly) looks up a bound
        # this run was actually judged against.
        "limits": recorded_limits,
        "producer": {
            "path": "ci/scripts/perf/profile_421_legs.sh",
            "kind": "script",
            "invocation": identity.get("producer_invocation"),
            "gating": "none",
        },
        "status": status,
        # The contract's validity gate tallied per leg (all four clauses) —
        # `status` above is the merge-level run status (its first three);
        # a headline "N of M legs pass the contract's validity gate" reads
        # THIS block, never `status`.
        "contract_validity": contract_validity,
        "notes": {
            "what": identity.get("what"),
            "gpu": identity.get("gpu"),
            "driver": identity.get("driver"),
            "cpu": identity.get("cpu"),
            "nsys": identity.get("nsys_human"),
            "checkpoints": checkpoints_notes,
            "recorded_deviations": recorded_deviations,
            # An operator-recorded fact with NO in-tree witness (e.g. an
            # off-pod disk-usage total): explicitly NOT covered by
            # `recorded_deviations`'s own "every number is filled from a
            # live source" guarantee — see `profile_421_run2_identity.json`'s
            # own header comment.
            "operator_recorded": identity.get("operator_recorded", {}),
        },
        "legs": legs,
        "p2": merge_report.get("p2_bf16", []),
        "attribution": attribution_report.get("legs", []),
        "realized_gains": attribution_report.get("realized_gains", []),
        "candidate_decisions": attribution_report.get("candidate_decisions", []),
        "findings": findings,
        "suppressed_findings": suppressed_findings,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="profile_421_artifact.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--legs-dir", required=True, help="a profile_421_legs.sh $OUT_DIR/legs to read manifests/census from")
    ap.add_argument(
        "--p2-dir",
        help=(
            "the PROFILE_421_P2_BF16=1 output dir ($OUT_DIR/p2-bf16), for the git_sha/box cross-check — "
            "REQUIRED (refused otherwise) whenever --merge-json carries any p2_bf16 rows"
        ),
    )
    ap.add_argument("--merge-json", required=True, help="profile_421_merge.py's own output over the SAME --legs-dir")
    ap.add_argument("--attribution-json", required=True, help="profile_421_attribute.py's own output over the SAME --legs-dir")
    ap.add_argument("--identity", required=True, help="the non-numeric identity/notes sidecar (see profile_421_run2_identity.json)")
    ap.add_argument("--out", help="write the artifact here (default: stdout)")
    args = ap.parse_args(argv)

    legs_dir = Path(args.legs_dir)
    p2_dir = Path(args.p2_dir) if args.p2_dir else None
    merge_json_path = Path(args.merge_json)
    attribution_json_path = Path(args.attribution_json)
    identity_path = Path(args.identity)
    try:
        merge_report = _load_json(merge_json_path, "--merge-json")
        attribution_report = _load_json(attribution_json_path, "--attribution-json")
        identity = _load_json(identity_path, "--identity")
        report = build_report(legs_dir, p2_dir, merge_report, attribution_report, identity)
        # Identity-completeness (never part of the hermetic `build_report`
        # core, which takes already-parsed dicts with no file identity of
        # their own): the sha256 of every INPUT FILE and of this producer's
        # OWN source (`SOURCE_FILES_FOR_NUMBERS`), so regeneration from the
        # SAME inputs against the SAME producer bytes is provably
        # byte-identical, and a later regeneration from a hand-edited input
        # or an edited producer is provably NOT (`check_cuda_run_
        # artifacts.py` recomputes `source_sha256` at its own HEAD and
        # refuses a mismatch by name).
        report["producer"]["invocation_argv"] = ["profile_421_artifact.py", *argv]
        # `producer.identity`: the fixed marker `check_cuda_run_
        # artifacts.py`'s rule (j) requires BOTH `source_sha256` and
        # `input_sha256` alongside, stamped BEFORE either so a mid-build
        # refusal below never leaves a half-stamped producer block on a
        # written artifact (`main` returns 1 before `args.out` is ever
        # touched on any `ArtifactBuildError`).
        report["producer"]["identity"] = PRODUCER_SOURCE_IDENTITY_MARKER
        report["producer"]["source_sha256"] = _source_sha256()
        p2_towers_for_manifest = p2_tower_names(merge_report)
        report["producer"]["input_sha256"] = {
            "merge_json": _sha256_file(merge_json_path, "--merge-json"),
            "attribution_json": _sha256_file(attribution_json_path, "--attribution-json"),
            "identity": _sha256_file(identity_path, "--identity"),
            **build_input_manifest(legs_dir, p2_dir, p2_towers_for_manifest),
        }
    except ArtifactBuildError as exc:
        print(f"::error::profile_421_artifact: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(report, indent=1, sort_keys=False)
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::profile_421_artifact: could not write {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)

    print(
        f"profile_421_artifact: git_sha={report['git_sha']} box={report['box']} "
        f"legs={len(report['legs'])} p2={len(report['p2'])} findings={len(report['findings'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
