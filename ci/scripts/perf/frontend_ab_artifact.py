#!/usr/bin/env python3
"""Issue #421 follow-on ("media front-end parallelization") close-out
artifact: turns `frontend_ab.sh`'s own raw legs + `frontend_ab_merge.py`'s
own merged `report.json` + a small non-numeric identity sidecar
(`frontend_ab_final_identity.json`) into the single committed
`crates/jammi-kernels/artifacts/cuda-runs/*.json`
`check_cuda_run_artifacts.py` schema-checks (contract `perf/421-frontend`
v3, "Build" item 1).

## Why re-derive rather than wrap `report.json`

`frontend_ab.sh` already writes its OWN trivial artifact wrapper (a
`{schema_version, git_sha, box, producer, status, report}` envelope around
whatever `frontend_ab_merge.py` happened to produce) -- see that script's
own tail. Wrapping a report is not proving one: nothing re-checks that the
wrapped `report.json` was actually produced by `frontend_ab_merge.py`'s own
pinned bar-decision rule from the raw legs sitting next to it, rather than
hand-edited or copied from a stale run. That gap is exactly the "restored
coverage 0.867 -> 0.895" transcription failure mode the house principles
name: a number could drift from its source, or never have come from a real
run at all, and a bare wrapper would read as equally authoritative either
way.

This module is the SECOND, INDEPENDENT reader: `build_report()` FIRST
enumerates `--raw-dir` itself (every `<tower>__<role>__<repeat>.{exit,json}`
stem actually present on disk) and refuses BY NAME unless that set equals
EXACTLY `TOWERS x ROLES x {r1..r<repeats>}` -- both directions: an extra
leg left over from a different `--repeats` value, or a leg the raw dir is
missing, is a refusal before a single number is read, never a silent
under- or over-count. The same exact-set discipline applies to
`--report-json`'s own `towers.<tower>.<role>` object: its repeat keys must
match `{r1..r<repeats>}` exactly too (a report claiming `repeats: 2` while
still carrying an `"r3"` entry is refused, not silently ignored). Only
THEN does it read every raw leg directly, re-derive `front_per_step`/
`train_per_step` per leg and the HTSAT bar's own
`p`/`ideal`/bounds/`ratio`/`ratio_lo`/`ratio_hi`/`verdict` from the SAME
pinned formula `frontend_ab_merge.py::build_report` uses (re-implemented
here, not imported, so the two readers cannot share a single bug) --
against `--serial-tail`'s own committed `task=... t_s=...` measurement line
(the run's own `frontend_serial_tail` example output, read verbatim off a
fixture file rather than transcribed onto a CLI flag) as WELL as
`report.json`'s own `serial_tail_ratio` -- and REFUSES (`ArtifactBuildError`)
the moment ANY re-derived number disagrees with `--report-json`'s own
value, or the two serial-tail-ratio variants disagree on the bar's own
VERDICT (the "driver default vs measured; verdict invariant" deviation
this contract names is a claim this module actually checks, not merely a
sentence a human typed). Every leg's own `tiers.finetune_run.task` and
`.device_name` are cross-checked against the tower being read and
`--report-json`'s own `box`, refusing by name on either mismatch. Both
serial-tail ratios (`r`) are domain-checked to `[0, 1)` at read time, and
every value this module would otherwise divide by is checked positive and
finite first -- a degenerate zero front-end time is a named refusal, never
a `ZeroDivisionError`.

## What comes from where

- Every per-leg `front_per_step`/`train_per_step`/`rayon_pool_threads` and
  the two `htsat_bar_*` blocks: computed HERE from `--raw-dir`'s own
  `provenance.build_sha`-cross-checked, exit-code-checked leg reports --
  never copied from `--report-json`, only cross-checked against it.
- `git_sha` (= `--report-json`'s own `tip_sha`), `box`, `base_sha`,
  `n_items_per_step`, `repeats`: read from `--report-json` (the ONE place
  the A/B driver itself recorded them) and used as inputs the raw legs are
  checked AGAINST (the declared build shas each leg's own
  `provenance.build_sha` must match).
- `notes.rendered_from_tree_sha` (`git rev-parse HEAD` at build time) and
  `notes.measured_tip_precedes_merge_tip.files_changed_since_measured_tip`
  / `.diffstat` (`git diff --name-only`/`--stat <tip_sha>
  <rendered_from_tree_sha>`): read live off `--repo-root`'s own git
  history, never hand-typed -- a branch tip named as "measured" can
  legitimately gain MORE commits before it merges (a concurrent fold
  landed on this very branch while this artifact was being built), so
  this module RECORDS the mechanical file list rather than refusing on
  it; which of those files sit on this unit's own timed front-end path is
  a judgement call a file-diff cannot make by itself (an "identical
  arithmetic, extracted into a helper" edit and a "new O(n) sequential
  pass added before the parallel stage" edit are NOT the same claim even
  though both touch the same file) -- `measured_tip_precedes_merge_tip.
  commentary` is that reviewed judgement, read from `--identity`
  (`measured_tip_precedes_merge_tip_commentary`), attached to the
  mechanically-computed file list rather than replacing it.
- `notes.input_sha256` / `notes.run_sha256`: sha256 digests computed live
  over the exact input bytes this run reads (`--report-json`, `--identity`,
  `--serial-tail`, and both the `.json` report AND the `.exit` code of
  every raw leg under `--raw-dir` -- `_read_raw_leg` reads and refuses on
  the `.exit` file just as much as the report, so it is hashed too), so a
  downstream reader can verify the committed artifact was rendered from the
  exact fixture bytes also committed alongside it.
- `verdict.serial_tail_ratio_deviation.r_measured`: `--serial-tail`'s own
  `task=audio_embedding ... t_s=...` line, divided by the re-derived HTSAT
  `front_base_mean_s` -- read from a committed fixture file (the job log's
  own serial-tail phase output), never a hand-typed CLI float.
- Everything else in `notes` (`what`, `gpu`, `driver`, `cpu`,
  `recorded_deviations`'s own static prose, `producer_invocation`): read
  from `--identity`, which carries ONLY provenance/prose that has no
  numeric guarantee attached (a device-model string, a shared-host-load
  observation) -- never a bare measurement this module could otherwise
  re-derive.

## The verdict

`verdict.unit_verdict` is never hardcoded: it is `htsat_bar_driver_r`'s own
re-derived `verdict` string, ASSERTED (refusing on disagreement, never
silently picking one) to equal `htsat_bar_measured_r`'s own verdict too --
the "verdict invariant" the contract names is a computed equality check,
not prose. `verdict.contract_clause_applied` is prose selected by that
computed verdict (PASS -> ACTIVATE; anything else -> the v3 contract's own
"ships without the efficiency claim" clause), never independently typed.

## Regeneration discipline: the artifact commit must be the unit's LAST commit

`notes.rendered_from_tree_sha` is `git rev-parse HEAD` taken AT BUILD TIME --
i.e. BEFORE the commit that adds or updates the committed artifact file
itself is made (this script only WRITES the file; something else commits
it afterwards). That ordering means the value this module records is
ALWAYS the artifact file's own eventual commit's PARENT, never that commit
itself, and it is a FIXPOINT: the moment any later commit lands on the
branch, the committed `files_changed_since_measured_tip` list (and,
transitively, `rendered_from_tree_sha`) is stale relative to a fresh
regeneration, through no fault of anything this module computed -- it was
correct at the instant it was rendered, and time moved on underneath it.

The discipline this buys is therefore procedural, not something this
script can enforce on itself at build time: **regenerating and committing
this artifact must be the LAST commit of whatever unit produces it.** A
gate exists to CATCH a violation of that discipline after the fact --
`test_frontend_ab_artifact.py`'s `assert_committed_artifact_not_stale`,
exercised by `RealFixtureRegressionTests.
test_committed_artifact_record_is_not_stale`. That gate is HEAD-independent
by construction: it never re-derives the record against whatever the
checkout's own `HEAD` happens to be (a PR's `refs/pull/N/merge` ref, or a
later merge to `main`, would otherwise name files that ARE this repo's own
unrelated churn as spurious staleness). Instead it has two arms -- (a) a
record check against `C` (the artifact file's own last commit, `git log -1
-- <artifact path>`) and `C`'s own parent, computed purely from git
history, ALWAYS; and (b), only when the caller knows the unit's own PR head
sha (CI wires this as `JAMMI_CI_UNIT_HEAD_SHA`, empty on a push-to-main
checkout), a check that `C` itself IS that head -- any later commit on the
PR, whether or not it touches a file the record already names, is then a
named failure, "artifact record stale: regenerate as the final commit",
never a silent pass.

Run: `python3 ci/scripts/perf/frontend_ab_artifact.py --raw-dir <dir>
--report-json <path> --identity <path> --serial-tail <path>
[--repo-root <dir>] [--out <path>]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
TOWERS = ("htsat", "clip-vision")
ROLES = ("base", "tip")
# The task each tower's own `tiers.finetune_run.task` must carry -- read
# straight off the raw leg and cross-checked against the tower this reader
# thinks it is reading (advisory: a leg copied into the wrong tower's own
# filename slot is a refusal, never a silently-mislabeled measurement).
TOWER_TASK = {"htsat": "audio_embedding", "clip-vision": "image_embedding"}
REDERIVE_REL_TOL = 1e-9
REDERIVE_ABS_TOL = 1e-12
# `r` (a serial-tail-time / front-end-time ratio) is a proper fraction by
# construction of the bar formula (`ideal = n / ceil(n/p)`, `bound = r +
# (1-r)/x`): `r == 1` degenerates every bound to exactly `1`, and `r > 1`
# or `r < 0` is not a ratio this formula was derived for at all -- both
# edges are refused, never silently accepted into arithmetic that would
# still "run" and produce a number.
R_DOMAIN_LO = 0.0
R_DOMAIN_HI = 1.0

_SERIAL_TAIL_LINE_RE = re.compile(r"^task=(?P<task>\S+)\s.*\bt_s=(?P<t_s>[^\s]+)\s*$")

# Same discipline as `check_cuda_run_artifacts.py`'s own `_run` helper: kill
# a background `git maintenance`/`gc --auto` writer at the source for every
# git invocation this module makes (the exact race that hit that gate's own
# tempdir teardown).
_GIT_NO_BACKGROUND_MAINTENANCE = ("-c", "gc.auto=0", "-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")


class ArtifactBuildError(Exception):
    """Uncomputable or inconsistent input -- fails closed, never guesses."""


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def _finite(value: object, label: str) -> float:
    """A float read that refuses non-finite input AT THE POINT OF READING.

    Every value this module compares or divides is routed through here
    first: an ordinary threshold or `min`/`max` call would let a diverged
    (`NaN`) upstream measurement sail through silently -- `NaN > 0`,
    `NaN < 0` and `NaN == 0` are ALL `False`.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactBuildError(f"{label} is not a number ({value!r})")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ArtifactBuildError(f"{label} is not finite ({value!r})")
    return as_float


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=REDERIVE_REL_TOL, abs_tol=REDERIVE_ABS_TOL):
        raise ArtifactBuildError(
            f"{label}: re-derived from --raw-dir as {actual!r}, but --report-json's own value is "
            f"{expected!r} -- disagreement of {abs(actual - expected)!r}"
        )


def _load_json(path: Path, what: str) -> dict:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {what} at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactBuildError(f"{what} at {path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ArtifactBuildError(f"{what} at {path} is not a JSON object")
    return loaded


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ArtifactBuildError(f"could not read {path} to hash it: {exc}") from exc


def _repeat_labels(repeats: int) -> tuple[str, ...]:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ArtifactBuildError(f"--report-json's own 'repeats' must be a positive int, got {repeats!r}")
    return tuple(f"r{i}" for i in range(1, repeats + 1))


def _validate_r(value: float, label: str) -> float:
    """`r` is a serial-tail/front-end time ratio the bar formula is only
    defined for on `[0, 1)` -- see the module-level `R_DOMAIN_LO`/`_HI`
    comment. Both edges (`r == 1`, `r > 1`, `r < 0`) are named refusals,
    never silently accepted into arithmetic that would still 'run'."""
    value = _finite(value, label)
    if not (R_DOMAIN_LO <= value < R_DOMAIN_HI):
        raise ArtifactBuildError(f"{label} must be in [{R_DOMAIN_LO}, {R_DOMAIN_HI}), got {value!r}")
    return value


def _require_positive_divisor(value: float, label: str) -> float:
    """Every value this module divides BY must be finite and strictly
    positive -- a degenerate zero (or negative, or non-finite) front-end
    measurement used as a divisor is a named refusal, never a
    `ZeroDivisionError` or a silently-propagated `NaN`/`inf`."""
    value = _finite(value, label)
    if value <= 0:
        raise ArtifactBuildError(f"{label} must be a finite positive number to divide by, got {value!r}")
    return value


def _validate_leg_set(raw_dir: Path, repeat_ids: tuple[str, ...]) -> None:
    """Enumerates `--raw-dir` itself (every `.json`/`.exit` file's own
    stem) and refuses BY NAME unless that set equals EXACTLY
    `TOWERS x ROLES x repeat_ids` -- in BOTH directions. This runs BEFORE
    a single leg is read: a stale `r3` pair left over from a `--repeats 3`
    run sitting next to a `report.json` that now declares `repeats: 2`
    would otherwise never be looked at (the per-leg reader below only ever
    asks for `repeat_ids`, so it cannot notice an EXTRA file on disk on
    its own)."""
    try:
        entries = list(raw_dir.iterdir())
    except OSError as exc:
        raise ArtifactBuildError(f"could not list --raw-dir at {raw_dir}: {exc}") from exc

    present_stems: set[str] = set()
    for entry in entries:
        if not entry.is_file() or entry.suffix not in (".json", ".exit"):
            continue
        present_stems.add(entry.stem)

    expected_stems = {f"{tower}__{role}__{repeat}" for tower in TOWERS for role in ROLES for repeat in repeat_ids}
    if present_stems != expected_stems:
        extra = sorted(present_stems - expected_stems)
        missing = sorted(expected_stems - present_stems)
        raise ArtifactBuildError(
            f"--raw-dir at {raw_dir} does not carry exactly TOWERS x ROLES x {{{', '.join(repeat_ids)}}} "
            f"(from --report-json's own 'repeats'={len(repeat_ids)}) -- extra leg stems: {extra!r}, "
            f"missing leg stems: {missing!r}"
        )


def _validate_towers_repeat_keys(towers: dict, repeat_ids: tuple[str, ...]) -> None:
    """The same exact-set discipline as `_validate_leg_set`, applied to
    `--report-json`'s own `towers.<tower>.<role>` object: a report
    claiming `repeats: 2` while still carrying a leftover `'r3'` entry (or
    missing one of `r1`/`r2`) is refused here, independent of whatever is
    sitting on `--raw-dir`."""
    for tower in TOWERS:
        by_tower = towers.get(tower)
        if not isinstance(by_tower, dict):
            raise ArtifactBuildError(f"--report-json's own towers.{tower} is not an object")
        for role in ROLES:
            by_role = by_tower.get(role)
            if not isinstance(by_role, dict):
                raise ArtifactBuildError(f"--report-json's own towers.{tower}.{role} is not an object")
            actual_keys = set(by_role.keys())
            expected_keys = set(repeat_ids)
            if actual_keys != expected_keys:
                raise ArtifactBuildError(
                    f"--report-json's own towers.{tower}.{role} does not carry exactly "
                    f"{{{', '.join(repeat_ids)}}} (from its own 'repeats'={len(repeat_ids)}) -- "
                    f"extra keys: {sorted(actual_keys - expected_keys)!r}, "
                    f"missing keys: {sorted(expected_keys - actual_keys)!r}"
                )


def _read_measured_serial_tail_s(serial_tail_path: Path, task: str) -> float:
    """Reads the run's own `cargo run -p jammi-bench --example
    frontend_serial_tail` measurement for `task` off a committed fixture
    file (the job log's own serial-tail phase output, verbatim) rather
    than a hand-typed CLI float. Named refusal if the file cannot be read,
    or if no `task=<task> ... t_s=<float>` line is present for `task`, or
    if that line's own `t_s` does not parse as a float."""
    try:
        text = serial_tail_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactBuildError(f"could not read --serial-tail at {serial_tail_path}: {exc}") from exc

    for line in text.splitlines():
        match = _SERIAL_TAIL_LINE_RE.match(line.strip())
        if match is None or match.group("task") != task:
            continue
        raw_t_s = match.group("t_s")
        try:
            return float(raw_t_s)
        except ValueError as exc:
            raise ArtifactBuildError(
                f"--serial-tail {serial_tail_path}: task={task} line's own t_s does not parse as a float "
                f"({raw_t_s!r})"
            ) from exc

    raise ArtifactBuildError(f"--serial-tail {serial_tail_path}: no 'task={task} ... t_s=...' line found")


# --------------------------------------------------------------------------- #
# git helpers -- mirrors `check_cuda_run_artifacts.py`'s own discipline
# --------------------------------------------------------------------------- #
def _run_git(args: list[str], repo_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *_GIT_NO_BACKGROUND_MAINTENANCE, *args], cwd=repo_root, capture_output=True, text=True
    )


def _is_shallow_repository(repo_root: Path) -> bool:
    proc = _run_git(["rev-parse", "--is-shallow-repository"], repo_root)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _rev_parse_head(repo_root: Path) -> str:
    proc = _run_git(["rev-parse", "HEAD"], repo_root)
    if proc.returncode != 0:
        raise ArtifactBuildError(f"`git rev-parse HEAD` failed in {repo_root}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _diff_name_only(a: str, b: str, repo_root: Path) -> set[str]:
    proc = _run_git(["diff", "--name-only", a, b], repo_root)
    if proc.returncode != 0:
        raise ArtifactBuildError(f"`git diff --name-only {a} {b}` failed in {repo_root}: {proc.stderr.strip()}")
    return {line for line in proc.stdout.splitlines() if line.strip()}


def _diff_stat(a: str, b: str, repo_root: Path) -> str:
    proc = _run_git(["diff", "--stat", a, b], repo_root)
    if proc.returncode != 0:
        raise ArtifactBuildError(f"`git diff --stat {a} {b}` failed in {repo_root}: {proc.stderr.strip()}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# raw-leg reading -- independent of `frontend_ab_merge.py::load_leg`
# --------------------------------------------------------------------------- #
def _read_raw_leg(raw_dir: Path, tower: str, role: str, repeat: str, declared_sha: str, declared_box: str) -> dict:
    leg_id = f"{tower}/{role}/{repeat}"
    exit_path = raw_dir / f"{tower}__{role}__{repeat}.exit"
    json_path = raw_dir / f"{tower}__{role}__{repeat}.json"

    if not exit_path.is_file():
        raise ArtifactBuildError(f"{leg_id}: missing .exit file at {exit_path}")
    exit_text = exit_path.read_text(encoding="utf-8").strip()
    try:
        exit_code = int(exit_text)
    except ValueError as exc:
        raise ArtifactBuildError(f"{leg_id}: .exit file {exit_path} does not carry an int ({exit_text!r})") from exc
    if exit_code != 0:
        raise ArtifactBuildError(f"{leg_id}: leg exited {exit_code} != 0 -- cannot re-derive a finding from a failed leg")

    if not json_path.is_file():
        raise ArtifactBuildError(f"{leg_id}: missing report file at {json_path}")
    report = _load_json(json_path, f"{leg_id} report")

    provenance = report.get("provenance")
    build_sha = provenance.get("build_sha") if isinstance(provenance, dict) else None
    if not isinstance(build_sha, str) or not build_sha:
        raise ArtifactBuildError(f"{leg_id}: report carries no provenance.build_sha")
    if build_sha != declared_sha:
        raise ArtifactBuildError(
            f"{leg_id}: report's own provenance.build_sha ({build_sha}) does not match the {role} sha "
            f"--report-json declares ({declared_sha}) -- this leg was not built from the sha it claims"
        )

    tiers = report.get("tiers")
    tier = tiers.get("finetune_run") if isinstance(tiers, dict) else None
    if not isinstance(tier, dict):
        raise ArtifactBuildError(f"{leg_id}: report carries no tiers.finetune_run object")

    steps = tier.get("steps_measured")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ArtifactBuildError(f"{leg_id}: steps_measured is not a positive int ({steps!r})")
    front_wall = _finite(tier.get("media_front_end_wall_s"), f"{leg_id}: media_front_end_wall_s")
    train_wall = _finite(tier.get("train_run_wall_s"), f"{leg_id}: train_run_wall_s")

    # Advisory: `task` (does this leg's own filename tower match what it
    # was actually run as?) and `device_name` (does it match the box
    # `--report-json` declares?) are cross-checked here, refusing by name
    # on either mismatch -- a leg copied into the wrong tower's own
    # filename slot, or run on a different box than the driver declares,
    # would otherwise read back as a silently-mislabeled measurement.
    task = tier.get("task")
    if not isinstance(task, str) or not task:
        raise ArtifactBuildError(f"{leg_id}: report carries no tiers.finetune_run.task")
    expected_task = TOWER_TASK[tower]
    if task != expected_task:
        raise ArtifactBuildError(
            f"{leg_id}: tiers.finetune_run.task is {task!r}, but tower {tower!r} expects {expected_task!r}"
        )
    device_name = tier.get("device_name")
    if not isinstance(device_name, str) or not device_name:
        raise ArtifactBuildError(f"{leg_id}: report carries no tiers.finetune_run.device_name")
    declared_box_device = declared_box.split(",")[0].strip()
    if device_name != declared_box_device:
        raise ArtifactBuildError(
            f"{leg_id}: tiers.finetune_run.device_name ({device_name!r}) does not match --report-json's own "
            f"box device prefix ({declared_box_device!r} of box {declared_box!r})"
        )

    rayon_pool_threads = tier.get("rayon_pool_threads")
    if role == "tip":
        if isinstance(rayon_pool_threads, bool) or not isinstance(rayon_pool_threads, int) or rayon_pool_threads <= 0:
            raise ArtifactBuildError(
                f"{leg_id}: tip leg carries no positive int rayon_pool_threads ({rayon_pool_threads!r})"
            )
    elif rayon_pool_threads is not None:
        raise ArtifactBuildError(
            f"{leg_id}: base leg (pre-#421-follow-on) unexpectedly carries rayon_pool_threads ({rayon_pool_threads!r})"
        )

    return {
        "leg_id": leg_id,
        "json_path": json_path,
        "exit_path": exit_path,
        "steps_measured": steps,
        "front_per_step": front_wall / steps,
        "train_per_step": train_wall / steps,
        "rayon_pool_threads": rayon_pool_threads,
    }


def _get_reported_leg(report_json: dict, tower: str, role: str, repeat: str) -> dict:
    leg_id = f"{tower}/{role}/{repeat}"
    towers = report_json.get("towers")
    if not isinstance(towers, dict):
        raise ArtifactBuildError("--report-json carries no 'towers' object")
    by_tower = towers.get(tower)
    by_role = by_tower.get(role) if isinstance(by_tower, dict) else None
    rj_leg = by_role.get(repeat) if isinstance(by_role, dict) else None
    if not isinstance(rj_leg, dict):
        raise ArtifactBuildError(f"{leg_id}: --report-json carries no towers.{tower}.{role}.{repeat} entry")
    if rj_leg.get("outcome") != "OK":
        raise ArtifactBuildError(f"{leg_id}: --report-json's own outcome is {rj_leg.get('outcome')!r}, not 'OK'")
    return rj_leg


# --------------------------------------------------------------------------- #
# the pinned bar-decision rule -- re-implemented (not imported) from
# `frontend_ab_merge.py::build_report`'s own arithmetic, so the two readers
# cannot share a single formula bug.
# --------------------------------------------------------------------------- #
def _compute_bar(front_by_role: dict[str, dict[str, float]], p: int, n: int, r: float) -> dict:
    r = _validate_r(r, "bar's own r")
    ideal = n / math.ceil(n / p)
    upper_bound = r + (1 - r) / (0.5 * ideal)
    lower_bound = r + (1 - r) / ideal

    tip_vals = list(front_by_role["tip"].values())
    base_vals = list(front_by_role["base"].values())
    max_base = _require_positive_divisor(max(base_vals), "front_per_step base values' own max (ratio_lo's own divisor)")
    min_base = _require_positive_divisor(min(base_vals), "front_per_step base values' own min (ratio_hi's own divisor)")
    front_tip = sum(tip_vals) / len(tip_vals)
    front_base = _require_positive_divisor(sum(base_vals) / len(base_vals), "front_base_mean_s (ratio's own divisor)")
    ratio = front_tip / front_base
    ratio_lo = min(tip_vals) / max_base
    ratio_hi = max(tip_vals) / min_base

    if ratio_hi <= upper_bound and ratio_lo >= lower_bound:
        verdict = "PASS"
    elif ratio_lo > upper_bound:
        verdict = "FAIL"
    elif ratio_hi < lower_bound:
        verdict = "INVALID_BEATS_IDEAL"
    else:
        verdict = "UNRESOLVED"

    return {
        "p": p,
        "n": n,
        "ideal": ideal,
        "r": r,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "front_tip_mean_s": front_tip,
        "front_base_mean_s": front_base,
        "ratio": ratio,
        "ratio_lo": ratio_lo,
        "ratio_hi": ratio_hi,
        "verdict": verdict,
    }


_BAR_FIELDS = (
    "p", "n", "ideal", "r", "upper_bound", "lower_bound",
    "front_tip_mean_s", "front_base_mean_s", "ratio", "ratio_lo", "ratio_hi",
)


# --------------------------------------------------------------------------- #
# top-level assembly
# --------------------------------------------------------------------------- #
def build_report(
    raw_dir: Path,
    report_json_path: Path,
    report_json: dict,
    identity_path: Path,
    identity: dict,
    serial_tail_path: Path,
    repo_root: Path,
    invocation: str,
) -> dict:
    for key in (
        "base_sha", "tip_sha", "box", "serial_tail_ratio", "n_items_per_step",
        "repeats", "status", "towers", "htsat_bar", "clip_vision_report_only",
    ):
        if key not in report_json:
            raise ArtifactBuildError(f"--report-json missing required key {key!r}")

    base_sha = report_json["base_sha"]
    tip_sha = report_json["tip_sha"]
    box = report_json["box"]
    if not isinstance(base_sha, str) or not base_sha:
        raise ArtifactBuildError(f"--report-json's own base_sha must be a non-empty string, got {base_sha!r}")
    if not isinstance(tip_sha, str) or not tip_sha:
        raise ArtifactBuildError(f"--report-json's own tip_sha must be a non-empty string, got {tip_sha!r}")
    if not isinstance(box, str) or not box:
        raise ArtifactBuildError(f"--report-json's own box must be a non-empty string, got {box!r}")

    r_driver = _validate_r(report_json["serial_tail_ratio"], "--report-json serial_tail_ratio")
    n = report_json["n_items_per_step"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ArtifactBuildError(f"--report-json's own n_items_per_step must be a positive int, got {n!r}")
    repeat_ids = _repeat_labels(report_json["repeats"])
    declared_sha_by_role = {"base": base_sha, "tip": tip_sha}

    towers_obj = report_json["towers"]
    if not isinstance(towers_obj, dict):
        raise ArtifactBuildError("--report-json's own towers is not an object")
    # ---- leg-set enumeration: refuse BY NAME before a single number is
    # read unless --raw-dir and --report-json's own towers object both
    # carry EXACTLY TOWERS x ROLES x repeat_ids -- neither an extra leg
    # left over from a different --repeats value nor a missing one can
    # slip past the per-repeat-id loop below on its own.
    _validate_leg_set(raw_dir, repeat_ids)
    _validate_towers_repeat_keys(towers_obj, repeat_ids)

    # ---- re-derive every leg directly from --raw-dir, cross-checked ----
    legs: dict[str, dict] = {}
    front_by_tower: dict[str, dict[str, dict[str, float]]] = {t: {"base": {}, "tip": {}} for t in TOWERS}
    for tower in TOWERS:
        for role in ROLES:
            for repeat in repeat_ids:
                leg = _read_raw_leg(raw_dir, tower, role, repeat, declared_sha_by_role[role], box)
                legs[f"{tower}__{role}__{repeat}"] = leg
                front_by_tower[tower][role][repeat] = leg["front_per_step"]

                rj_leg = _get_reported_leg(report_json, tower, role, repeat)
                if leg["steps_measured"] != rj_leg.get("steps_measured"):
                    raise ArtifactBuildError(
                        f"{leg['leg_id']}: re-derived steps_measured {leg['steps_measured']!r} does not match "
                        f"--report-json's own {rj_leg.get('steps_measured')!r}"
                    )
                _assert_close(
                    leg["front_per_step"], _finite(rj_leg.get("front_per_step"), f"{leg['leg_id']}: front_per_step"),
                    f"{leg['leg_id']} front_per_step",
                )
                _assert_close(
                    leg["train_per_step"], _finite(rj_leg.get("train_per_step"), f"{leg['leg_id']}: train_per_step"),
                    f"{leg['leg_id']} train_per_step",
                )
                if leg["rayon_pool_threads"] != rj_leg.get("rayon_pool_threads"):
                    raise ArtifactBuildError(
                        f"{leg['leg_id']}: re-derived rayon_pool_threads {leg['rayon_pool_threads']!r} does not "
                        f"match --report-json's own {rj_leg.get('rayon_pool_threads')!r}"
                    )

    tip_p_values = {legs[f"htsat__tip__{repeat}"]["rayon_pool_threads"] for repeat in repeat_ids}
    if len(tip_p_values) != 1:
        raise ArtifactBuildError(f"htsat tip legs disagree on rayon_pool_threads: {sorted(tip_p_values)!r}")
    p = next(iter(tip_p_values))

    # ---- re-derive the HTSAT bar (driver-default r), cross-check verbatim ----
    htsat_bar_driver = _compute_bar(front_by_tower["htsat"], p, n, r_driver)
    rj_bar = report_json["htsat_bar"]
    if not isinstance(rj_bar, dict):
        raise ArtifactBuildError("--report-json's own htsat_bar is not an object")
    for field in _BAR_FIELDS:
        _assert_close(
            htsat_bar_driver[field], _finite(rj_bar.get(field), f"--report-json htsat_bar.{field}"),
            f"htsat_bar.{field}",
        )
    if htsat_bar_driver["verdict"] != rj_bar.get("verdict"):
        raise ArtifactBuildError(
            f"re-derived htsat_bar verdict {htsat_bar_driver['verdict']!r} disagrees with --report-json's own "
            f"{rj_bar.get('verdict')!r}"
        )

    # ---- the run's OWN measured serial-tail ratio, read off --serial-tail
    # (a committed fixture carrying the job log's own verbatim
    # 'task=... t_s=...' line), same pinned rule ----
    measured_serial_tail_s = _read_measured_serial_tail_s(serial_tail_path, TOWER_TASK["htsat"])
    front_base_mean_s = _require_positive_divisor(
        htsat_bar_driver["front_base_mean_s"], "htsat_bar_driver's own front_base_mean_s (r_measured's own divisor)"
    )
    r_measured = _validate_r(measured_serial_tail_s / front_base_mean_s, "r_measured (--serial-tail / front_base_mean_s)")
    htsat_bar_measured = _compute_bar(front_by_tower["htsat"], p, n, r_measured)

    if htsat_bar_driver["verdict"] != htsat_bar_measured["verdict"]:
        raise ArtifactBuildError(
            f"the 'driver default vs measured; verdict invariant' deviation does not hold on this run: "
            f"r_driver={r_driver!r} -> verdict {htsat_bar_driver['verdict']!r}, but "
            f"r_measured={r_measured!r} -> verdict {htsat_bar_measured['verdict']!r} -- re-measure before "
            "recording this as a deviation"
        )
    unit_verdict = htsat_bar_driver["verdict"]

    # ---- clip-vision report-only re-derivation ----
    rj_cv = report_json["clip_vision_report_only"]
    if not isinstance(rj_cv, dict):
        raise ArtifactBuildError("--report-json's own clip_vision_report_only is not an object")
    cv_tip_vals = list(front_by_tower["clip-vision"]["tip"].values())
    cv_base_vals = list(front_by_tower["clip-vision"]["base"].values())
    cv_front_tip = sum(cv_tip_vals) / len(cv_tip_vals)
    cv_front_base = _require_positive_divisor(
        sum(cv_base_vals) / len(cv_base_vals), "clip_vision_report_only.front_base_mean_s (ratio's own divisor)"
    )
    cv_ratio = cv_front_tip / cv_front_base
    _assert_close(
        cv_front_tip, _finite(rj_cv.get("front_tip_mean_s"), "clip_vision_report_only.front_tip_mean_s"),
        "clip_vision_report_only.front_tip_mean_s",
    )
    _assert_close(
        cv_front_base, _finite(rj_cv.get("front_base_mean_s"), "clip_vision_report_only.front_base_mean_s"),
        "clip_vision_report_only.front_base_mean_s",
    )
    _assert_close(cv_ratio, _finite(rj_cv.get("ratio"), "clip_vision_report_only.ratio"), "clip_vision_report_only.ratio")

    # ---- git facts: rendered-from tree sha + the "measured tip precedes
    # merge tip" deviation. A branch tip that was "measured" can legitimately
    # gain MORE commits before it merges -- this module RECORDS the
    # mechanical file list (never refusing on it: a file-diff alone cannot
    # judge whether an edit changed timed behaviour) and attaches the
    # reviewed commentary --identity carries for exactly this deviation.
    if _is_shallow_repository(repo_root):
        raise ArtifactBuildError(
            "shallow checkout at --repo-root -- cannot compute the post-tip diff; use fetch-depth: 0"
        )
    rendered_from_tree_sha = _rev_parse_head(repo_root)
    post_tip_files = sorted(_diff_name_only(tip_sha, rendered_from_tree_sha, repo_root))
    post_tip_diffstat = _diff_stat(tip_sha, rendered_from_tree_sha, repo_root)
    measured_tip_precedes_merge_tip = {
        "kind": "measured-tip-precedes-merge-tip",
        "measured_tip": tip_sha,
        "rendered_from_tree_sha": rendered_from_tree_sha,
        "files_changed_since_measured_tip": post_tip_files,
        "diffstat": post_tip_diffstat,
        "commentary": identity.get("measured_tip_precedes_merge_tip_commentary"),
    }

    # ---- input hashes + a single run fingerprint over every raw leg read ----
    # Both files this module actually reads per leg -- the `.json` report AND
    # the `.exit` code (the leg-outcome gate at the top of `_read_raw_leg`
    # reads and refuses on the latter just as much as it reads the former,
    # so a downstream reader verifying "this artifact was rendered from
    # exactly these committed bytes" needs both hashed, not only the report).
    input_sha256 = {
        "report_json": _sha256_file(report_json_path),
        "identity": _sha256_file(identity_path),
        "serial_tail": _sha256_file(serial_tail_path),
    }
    for key, leg in sorted(legs.items()):
        input_sha256[f"raw/{key}.json"] = _sha256_file(leg["json_path"])
        input_sha256[f"raw/{key}.exit"] = _sha256_file(leg["exit_path"])
    run_sha256 = hashlib.sha256(
        b"".join(
            f"{key}\n".encode("utf-8") + legs[key]["exit_path"].read_bytes() + legs[key]["json_path"].read_bytes()
            for key in sorted(legs)
        )
    ).hexdigest()

    if unit_verdict == "PASS":
        contract_clause = (
            "ACTIVATE (keep the change): the HTSAT bar holds under both the driver-default and the run's own "
            "measured serial-tail ratio."
        )
    else:
        contract_clause = (
            f"not ACTIVATE (HTSAT bar {unit_verdict} under both the driver-default and the run's own measured "
            "serial-tail ratio); per contract v3's own Verdict clause the unit ships because bit identity holds "
            "and there is no serving regression, with the numbers recorded and NO efficiency claim."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": tip_sha,
        "box": box,
        "producer": {
            "path": "ci/scripts/perf/frontend_ab_artifact.py",
            "kind": "script",
            "invocation": invocation,
            "gating": "none",
        },
        "status": report_json["status"],
        "notes": {
            "what": identity.get("what"),
            "gpu": identity.get("gpu"),
            "driver": identity.get("driver"),
            "cpu": identity.get("cpu"),
            "driver_commit": tip_sha,
            "run_sha256": run_sha256,
            "rendered_from_tree_sha": rendered_from_tree_sha,
            "input_sha256": input_sha256,
            "measured_tip_precedes_merge_tip": measured_tip_precedes_merge_tip,
            "recorded_deviations": list(identity.get("recorded_deviations", [])),
        },
        "measurement": {
            "base_sha": base_sha,
            "tip_sha": tip_sha,
            "box": box,
            "n_items_per_step": n,
            "repeats": report_json["repeats"],
            "legs": {
                key: {
                    "front_per_step": legs[key]["front_per_step"],
                    "train_per_step": legs[key]["train_per_step"],
                    "steps_measured": legs[key]["steps_measured"],
                    "rayon_pool_threads": legs[key]["rayon_pool_threads"],
                }
                for key in sorted(legs)
            },
            "htsat_bar_driver_r": htsat_bar_driver,
            "htsat_bar_measured_r": htsat_bar_measured,
            "clip_vision_report_only": {
                "front_tip_mean_s": cv_front_tip,
                "front_base_mean_s": cv_front_base,
                "ratio": cv_ratio,
            },
        },
        "verdict": {
            "unit_verdict": unit_verdict,
            "cross_checked_against_report_json_htsat_bar_verdict": rj_bar.get("verdict"),
            "serial_tail_ratio_deviation": {
                "kind": "driver default vs measured; verdict invariant",
                "r_driver": r_driver,
                "r_driver_source": "FRONTEND_AB_SERIAL_TAIL_RATIO (rehearsal-derived default, contract v3)",
                "r_measured": r_measured,
                "r_measured_source": (
                    "this run's own 'cargo run -p jammi-bench --example frontend_serial_tail' measurement, read "
                    f"off --serial-tail's own 'task={TOWER_TASK['htsat']} ... t_s=...' line (a committed fixture "
                    "carrying the job log's own verbatim serial-tail phase output), divided by the re-derived "
                    "HTSAT front_base_mean_s"
                ),
                "verdict_under_r_driver": htsat_bar_driver["verdict"],
                "verdict_under_r_measured": htsat_bar_measured["verdict"],
                "invariant_holds": True,
            },
            "contract_clause_applied": contract_clause,
        },
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="frontend_ab_artifact.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-dir", required=True, help="the frontend_ab.sh $OUT_DIR/raw directory")
    ap.add_argument("--report-json", required=True, help="frontend_ab_merge.py's own merged report.json over the SAME --raw-dir")
    ap.add_argument("--identity", required=True, help="the non-numeric identity/notes sidecar (see frontend_ab_final_identity.json)")
    ap.add_argument(
        "--serial-tail", required=True,
        help=(
            "path to a fixture carrying this run's own 'cargo run -p jammi-bench --example "
            "frontend_serial_tail' output verbatim (a 'task=... t_s=...' line per task; see "
            "fixtures/frontend_ab_final/serial_tail.txt)"
        ),
    )
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="git repo root (default: this checkout's own root)")
    ap.add_argument("--out", help="write the artifact here (default: stdout)")
    args = ap.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    report_json_path = Path(args.report_json)
    identity_path = Path(args.identity)
    serial_tail_path = Path(args.serial_tail)
    repo_root = Path(args.repo_root)
    invocation = "python3 ci/scripts/perf/frontend_ab_artifact.py " + " ".join(argv)

    try:
        report_json = _load_json(report_json_path, "--report-json")
        identity = _load_json(identity_path, "--identity")
        report = build_report(
            raw_dir, report_json_path, report_json, identity_path, identity,
            serial_tail_path, repo_root, invocation,
        )
    except ArtifactBuildError as exc:
        print(f"::error::frontend_ab_artifact: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(report, indent=1, sort_keys=False)
    if args.out:
        try:
            Path(args.out).write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"::error::frontend_ab_artifact: could not write {args.out}: {exc}", file=sys.stderr)
            return 1
    else:
        print(payload)

    print(
        f"frontend_ab_artifact: git_sha={report['git_sha']} box={report['box']} "
        f"unit_verdict={report['verdict']['unit_verdict']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
