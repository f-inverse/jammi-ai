#!/usr/bin/env python3
"""Hermetic tests for `frontend_ab_artifact.py` (issue #421 follow-on
close-out artifact producer; contract `perf/421-frontend` v3).

Everything here is SYNTHETIC: a small `git init`'d tempdir repo (mirroring
`check_cuda_run_artifacts.py`'s own self-test discipline) stands in for
`--repo-root`, and every raw leg / `report.json` / identity sidecar is
hand-built in a tempdir -- no GPU, no pod, no network, and none of the real
pod-p421c measurement (that lives in the small COMMITTED fixture directory
`fixtures/frontend_ab_final/`, exercised separately by
`RealFixtureRegressionTests` below, never by this suite's own synthetic
classes).

`_oracle_bar` below is an INDEPENDENT re-implementation of the pinned bar
formula (a THIRD copy, after `frontend_ab_merge.py`'s own and
`frontend_ab_artifact.py`'s own) -- the numpy-first-oracle doctrine scaled
to plain-float arithmetic, per `profile_421_artifact.py`'s own test-suite
precedent ("the arithmetic here is simple enough that a second float
computation IS the independent check"). Every numeric field this suite
checks is checked against THIS oracle, never against a value copied out of
the module under test.

`LegSetEnumerationTests` drives the audit's own probe: a `--repeats`
shrink (3 -> 2) with the `r3` raw-leg files still present on disk (a stale
re-run) must REFUSE, never silently read only `r1`/`r2` and PASS -- in
both directions (extra legs, missing legs), on both `--raw-dir` and
`--report-json`'s own `towers` object independently. `RDomainTests`/
`RDomainEndToEndTests` cover `r`'s own `[0, 1)` domain at both edges (and
beyond), plus every division this module performs being guarded against a
zero/non-positive divisor (a named refusal, never a `ZeroDivisionError`).
`SerialTailFixtureTests` covers `--serial-tail`'s own missing-file/
missing-line/malformed-`t_s` refusals; `TowerTaskAndDeviceNameTests` covers
the per-leg `task`/`device_name` cross-check.

## The "measured tip precedes merge tip" record is a FIXPOINT of `C`, never of the checkout's own HEAD

`notes.measured_tip_precedes_merge_tip` (and the top-level
`notes.rendered_from_tree_sha` it mirrors) is only ever correct AT THE
INSTANT it is rendered -- `frontend_ab_artifact.py`'s own "Regeneration
discipline" module-doc section names why: the producer records `git
rev-parse HEAD` BEFORE the commit that adds/updates the artifact file
itself (call that commit `C`) lands, so the recorded value is always `C`'s
own PARENT, never `C` itself. The house discipline is therefore:
regenerating and committing this artifact is the unit's own LAST commit.

The mechanical gate for that discipline must itself be a property of `C`
and of the unit's own PR head -- NEVER of whatever the checkout's own
`HEAD` happens to be at gate time. A checkout at `refs/pull/N/merge` (CI's
own default PR ref, a synthetic merge of the PR branch onto its target)
or at `main` after the merge both carry files that changed on the OTHER
side of that merge -- re-deriving the record against such a `HEAD` would
name those unrelated files as "staleness", an unfixable red with no
regeneration able to clear it. `assert_committed_artifact_not_stale` below
is therefore split into two independent arms:

- **Arm (a), always run:** given `C` (the artifact file's own last commit,
  `git log -1 -- <path>`) and `C^` (its own parent), computed purely from
  git history: the committed `rendered_from_tree_sha` must equal `C^`
  exactly; `files_changed_since_measured_tip` must equal `git diff
  --name-only <measured_tip> C^` exactly (not a union with anything else);
  `diffstat` must equal `git diff --stat <measured_tip> C^`, normalised
  the same way the producer normalises it; the `measurement` block must be
  byte-identical to a fresh regeneration (numbers never move); and every
  OTHER field must also be byte-identical. None of this touches the
  checkout's own `HEAD` at all, so it is unaffected by whatever ref
  happens to be checked out.
- **Arm (b), event-keyed, never a silent empty-string skip:** CI wires
  `JAMMI_CI_UNIT_HEAD_SHA: ${{ github.event.pull_request.head.sha ||
  'push' }}` -- the PR's own head sha on a `pull_request` (or
  `pull_request_target`) checkout, the literal string `push` on a `push`
  checkout, never empty on either. `resolve_unit_head_sha` cross-checks
  the raw value against `GITHUB_EVENT_NAME` itself (a GitHub Actions
  built-in, present on every job with no extra wiring needed): on
  `pull_request`/`pull_request_target`, the `push` sentinel is REFUSED
  naming both `GITHUB_EVENT_NAME` and `JAMMI_CI_UNIT_HEAD_SHA` (the PR's
  own head sha must have been exported); on `push`, any value other than
  the sentinel is REFUSED as inconsistent, naming both variables. Past
  that cross-check, the raw env var is turned into either a full commit
  sha (arm (b) engages) or `None` (arm (b) is a deliberate no-op): the
  literal `push` resolves to `None` with a printed notice (the
  intentional push-to-main skip); an abbreviated sha or a ref name
  resolves to its full sha via `git rev-parse --verify <v>^{commit}` (so
  arm (b) compares full-sha-to-full-sha); a value `git` cannot resolve is
  a named `AssertionError` citing the malformed value itself -- NEVER
  diagnosed as staleness. Locally (no `GITHUB_EVENT_NAME` and no
  `JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED=1`), the event cross-check
  never engages and an empty/unset var resolves to `None` with a printed
  notice. Under `JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED=1` (CI), an
  empty/unset var is instead a named `AssertionError` -- the matrix's own
  `|| 'push'` fallback means empty can only be a wiring break, never a
  legitimate checkout of either event (this repo's own
  zero-execution-is-RED doctrine: a stale artifact with no working env
  var must never silently pass). Once resolved to a sha, arm (b) asserts
  `C` itself equals that sha: any later commit on the PR -- whether or
  not it touches a file the record already names -- is a named failure,
  since arm (a) alone cannot see a commit that never touched the artifact
  file itself.

Both arms raise the same named `AssertionError`, "artifact record stale:
regenerate as the final commit", never a silent pass.
`ArtifactRecordFreshnessGateTests` proves both arms on a scratch `git
init`'d repo (six scenarios: final-commit-with-matching-head green; a
later commit touching an already-listed file red under (b); a later
commit touching a brand-new file red under (b); a later merge commit
present with no PR head known still green under (a) alone; a two-parent
merge commit sitting AT `HEAD` with no PR head known still green under
(a); and a tampered record -- a file dropped from the committed list --
red under (a) even with no later commits at all).
`UnitHeadShaResolutionTests` covers `resolve_unit_head_sha`'s own
normalisation of the raw `JAMMI_CI_UNIT_HEAD_SHA` env var (the `push`
sentinel, an empty value under CI vs a bare local run, an unresolvable
value, an abbreviated sha, and the `GITHUB_EVENT_NAME` cross-check
against the sentinel in both directions) on its own scratch repo,
independent of `ArtifactRecordFreshnessGateTests`'s repo.
`RealFixtureRegressionTests.test_committed_artifact_record_is_not_stale`
then drives arm (a) against the REAL committed pod-p421c artifact, and
arm (b) via `resolve_unit_head_sha` whenever `JAMMI_CI_UNIT_HEAD_SHA` is
set in the environment.

Run: `python3 ci/scripts/perf/test_frontend_ab_artifact.py`
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PERF_DIR = Path(__file__).resolve().parent
ARTIFACT = PERF_DIR / "frontend_ab_artifact.py"
REAL_FIXTURE_DIR = PERF_DIR / "fixtures" / "frontend_ab_final"

sys.path.insert(0, str(PERF_DIR))
import frontend_ab_artifact as art  # noqa: E402

TOWERS = ("htsat", "clip-vision")
ROLES = ("base", "tip")
REPEATS = ("r1", "r2", "r3")
N_ITEMS_PER_STEP = 24
P = 8
R_DRIVER = 0.0
MEASURED_SERIAL_TAIL_S = 0.05  # -> r_measured = 0.05 / front_base_mean_s(=1.0) = 0.05
BOX = "TESTBOX"
DEVICE_NAME = "TESTBOX"  # -- box.split(',')[0].strip(): the per-leg device_name cross-check
TASK_BY_TOWER = {"htsat": "audio_embedding", "clip-vision": "image_embedding"}

# front_per_step per (tower, role) -- identical across repeats (spread 0)
# keeps the oracle arithmetic trivial to hand-check; the "spread" family is
# covered separately below (SpreadTests).
FRONT_PER_STEP = {
    ("htsat", "base"): 1.0,
    ("htsat", "tip"): 0.2,
    ("clip-vision", "base"): 0.5,
    ("clip-vision", "tip"): 0.1,
}
TRAIN_PER_STEP = {
    ("htsat", "base"): 1.2,
    ("htsat", "tip"): 0.4,
    ("clip-vision", "base"): 0.6,
    ("clip-vision", "tip"): 0.2,
}
STEPS = 100


def _oracle_bar(front_base_vals: list[float], front_tip_vals: list[float], p: int, n: int, r: float) -> dict:
    """Independent re-implementation of the pinned bar formula -- never
    calls `frontend_ab_artifact.py` or `frontend_ab_merge.py`."""
    ideal = n / math.ceil(n / p)
    upper_bound = r + (1 - r) / (0.5 * ideal)
    lower_bound = r + (1 - r) / ideal
    front_tip = sum(front_tip_vals) / len(front_tip_vals)
    front_base = sum(front_base_vals) / len(front_base_vals)
    ratio = front_tip / front_base
    ratio_lo = min(front_tip_vals) / max(front_base_vals)
    ratio_hi = max(front_tip_vals) / min(front_base_vals)
    if ratio_hi <= upper_bound and ratio_lo >= lower_bound:
        verdict = "PASS"
    elif ratio_lo > upper_bound:
        verdict = "FAIL"
    elif ratio_hi < lower_bound:
        verdict = "INVALID_BEATS_IDEAL"
    else:
        verdict = "UNRESOLVED"
    return {
        "p": p, "n": n, "ideal": ideal, "r": r,
        "upper_bound": upper_bound, "lower_bound": lower_bound,
        "front_tip_mean_s": front_tip, "front_base_mean_s": front_base,
        "ratio": ratio, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# a tiny `git init`'d tempdir repo -- mirrors `check_cuda_run_artifacts.py`'s
# own self-test discipline. Two commits: `base_sha` (touches `unit/hot.rs`
# for the first time) and `tip_sha` (modifies it again) -- these two shas
# are what every fixture's own `base_sha`/`tip_sha` + every raw leg's own
# `provenance.build_sha` are set to, so `frontend_ab_artifact.py`'s git
# calls resolve against REAL commits, never a fabricated string.
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {
        "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"git {args} failed: {proc.stderr}")
    return proc


# --------------------------------------------------------------------------- #
# the "measured tip precedes merge tip" record is a FIXPOINT of `C` (the
# artifact file's own last commit) -- see this module's own doc and
# `frontend_ab_artifact.py`'s own "Regeneration discipline" section.
# `assert_committed_artifact_not_stale` is the mechanical gate, split into
# arm (a) (HEAD-independent, always run) and arm (b) (only when the unit's
# own PR head sha is known); the small `git` helpers below give arm (a)
# exactly the facts it needs about `C` and `C^`, computed directly from git
# history rather than from whatever the checkout's own `HEAD` is.
# --------------------------------------------------------------------------- #
STALE_ARTIFACT_RECORD_MESSAGE = "artifact record stale: regenerate as the final commit"

# `ci.yml`'s own matrix entry exports `JAMMI_CI_UNIT_HEAD_SHA: ${{
# github.event.pull_request.head.sha || 'push' }}` -- see `resolve_unit_
# head_sha` below for the full normalisation contract.
_UNIT_HEAD_ENV_VAR = "JAMMI_CI_UNIT_HEAD_SHA"
_PUSH_SENTINEL = "push"
# GitHub Actions' own built-in env var, present on every job with no extra
# wiring -- `resolve_unit_head_sha` cross-checks it against `_PUSH_SENTINEL`
# so the sentinel and the event that produced it can never disagree.
_EVENT_NAME_ENV_VAR = "GITHUB_EVENT_NAME"
_PR_EVENT_NAMES = ("pull_request", "pull_request_target")

# The `notes.measured_tip_precedes_merge_tip` keys arm (a) checks against
# `C`/`C^` directly (never against a fresh-at-HEAD regeneration's own
# values for these three -- see `assert_committed_artifact_record_matches_
# artifact_commit`). Every other field is asserted byte-identical by
# `_report_sans_freshness_fields`'s own caller.
_MEASURED_TIP_DEV_VOLATILE_KEYS = ("rendered_from_tree_sha", "files_changed_since_measured_tip", "diffstat")


def _artifact_own_last_commit(artifact_rel_path: str, repo_root: Path) -> str:
    """`C`: the commit that last touched `artifact_rel_path` (`git log -1
    --format=%H -- <path>`) -- the SAME discriminator the module doc names.
    """
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", artifact_rel_path],
        cwd=repo_root, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git log -1 --format=%H -- {artifact_rel_path} failed in {repo_root}: {proc.stderr}")
    sha = proc.stdout.strip()
    if not sha:
        raise RuntimeError(f"no commit touches {artifact_rel_path} in {repo_root} -- is it tracked?")
    return sha


def _commit_parent(sha: str, repo_root: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", f"{sha}^"], cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git rev-parse {sha}^ failed in {repo_root}: {proc.stderr}")
    return proc.stdout.strip()


def _report_sans_freshness_fields(report: dict) -> dict:
    """A deep copy of `report` with every field this gate KNOWINGLY allows
    to move between a committed record and a fresh regeneration (the
    top-level `notes.rendered_from_tree_sha` mirror, plus
    `_MEASURED_TIP_DEV_VOLATILE_KEYS` inside `notes.
    measured_tip_precedes_merge_tip`) stripped out -- so a plain equality
    check on the remainder catches every OTHER divergence, by name, rather
    than requiring a bespoke per-field comparison a future field addition
    could silently bypass.
    """
    out = json.loads(json.dumps(report))
    notes = out.get("notes", {})
    notes.pop("rendered_from_tree_sha", None)
    dev = notes.get("measured_tip_precedes_merge_tip")
    if isinstance(dev, dict):
        for key in _MEASURED_TIP_DEV_VOLATILE_KEYS:
            dev.pop(key, None)
    return out


def assert_committed_artifact_record_matches_artifact_commit(
    committed: dict, regenerated: dict, artifact_rel_path: str, repo_root: Path
) -> None:
    """Arm (a): a check against `C` (`artifact_rel_path`'s own last commit)
    and `C^` (its own parent), computed PURELY from git history -- never
    against the checkout's own current `HEAD`. This is what makes the
    check safe to run on a PR's `refs/pull/N/merge` ref, or on `main` after
    the merge: `C`/`C^` are fixed commits regardless of what else is
    checked out alongside them.

    `committed` is the artifact JSON as it sits in the working tree;
    `regenerated` is a FRESH `build_report(...)` call made at `repo_root`'s
    CURRENT checkout, using the SAME recorded inputs/invocation -- used
    ONLY for the `measurement` block and the "every other field" check,
    both of which are independent of git position (the fixture bytes those
    fields are derived from do not change across a merge that never
    touches them). Raises `AssertionError` (never returns a bool) on any
    divergence.
    """
    c = _artifact_own_last_commit(artifact_rel_path, repo_root)
    c_parent = _commit_parent(c, repo_root)

    committed_dev = committed["notes"]["measured_tip_precedes_merge_tip"]

    if committed["notes"]["rendered_from_tree_sha"] != committed_dev["rendered_from_tree_sha"]:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: committed notes.rendered_from_tree_sha disagrees with "
            "notes.measured_tip_precedes_merge_tip.rendered_from_tree_sha within the SAME committed artifact "
            "-- this is not a valid frontend_ab_artifact.py output at all"
        )
    if committed_dev["rendered_from_tree_sha"] != c_parent:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: committed rendered_from_tree_sha "
            f"{committed_dev['rendered_from_tree_sha']!r} != {artifact_rel_path}'s own last commit ({c})'s "
            f"own parent ({c_parent!r})"
        )

    measured_tip = committed_dev["measured_tip"]
    expected_files = sorted(art._diff_name_only(measured_tip, c_parent, repo_root))
    committed_files = sorted(committed_dev["files_changed_since_measured_tip"])
    if committed_files != expected_files:
        extra = sorted(set(committed_files) - set(expected_files))
        missing = sorted(set(expected_files) - set(committed_files))
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: committed files_changed_since_measured_tip does not equal "
            f"`git diff --name-only {measured_tip} {c_parent}` (computed at {artifact_rel_path}'s own last "
            f"commit {c}'s own parent, never at the checkout's own HEAD) -- in the committed record but not "
            f"expected: {extra!r}; expected but absent from the committed record: {missing!r}"
        )

    expected_diffstat = art._diff_stat(measured_tip, c_parent, repo_root).strip()
    committed_diffstat = committed_dev["diffstat"].strip()
    if committed_diffstat != expected_diffstat:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: committed diffstat does not equal `git diff --stat {measured_tip} "
            f"{c_parent}` (computed at {artifact_rel_path}'s own last commit {c}'s own parent)"
        )

    committed_measurement = json.dumps(committed["measurement"], sort_keys=True)
    regenerated_measurement = json.dumps(regenerated["measurement"], sort_keys=True)
    if committed_measurement != regenerated_measurement:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: the committed measurement block is not byte-identical to a "
            "fresh regeneration -- a number moved"
        )

    committed_rest = _report_sans_freshness_fields(committed)
    regenerated_rest = _report_sans_freshness_fields(regenerated)
    if committed_rest != regenerated_rest:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: fields outside notes.rendered_from_tree_sha / "
            "notes.measured_tip_precedes_merge_tip.{rendered_from_tree_sha,files_changed_since_measured_tip,"
            "diffstat} / measurement diverge between the committed artifact and a fresh regeneration"
        )


def resolve_unit_head_sha(
    raw_value: str | None, repo_root: Path, *, ci_mode: bool, event_name: str | None = None
) -> str | None:
    """Normalises the raw `JAMMI_CI_UNIT_HEAD_SHA` env var into either a
    full commit sha (arm (b), `assert_artifact_commit_is_unit_head`,
    engages) or `None` (arm (b) is a deliberate no-op) -- reshaped to this
    repo's own zero-execution-is-RED doctrine so an empty expansion under
    CI can never silently turn arm (b) off.

    `ci.yml`'s own matrix entry exports `JAMMI_CI_UNIT_HEAD_SHA: ${{
    github.event.pull_request.head.sha || 'push' }}`: on a `pull_request`
    (or `pull_request_target`) checkout that is the PR's own head sha,
    NEVER empty and NEVER the literal `push`; on a `push` checkout it is
    the literal string `push`, NEVER empty and NEVER a resolved sha.
    `event_name` (CI passes `GITHUB_EVENT_NAME`, a GitHub Actions built-in
    present on every job with no extra wiring) lets this function cross-
    check the two variables against each other rather than trusting
    `raw_value` alone. Six cases, checked in this order:

    - `event_name` is `pull_request`/`pull_request_target` and `raw_value`
      is the literal `push`: REFUSED naming both `GITHUB_EVENT_NAME` and
      `JAMMI_CI_UNIT_HEAD_SHA` -- a pull-request checkout's own head sha
      must have been exported; the sentinel here can only mean the
      matrix's own expression broke.
    - `event_name` is `push` and `raw_value` is a non-empty value other
      than the literal `push`: REFUSED as inconsistent, naming both
      variables -- a push-to-main checkout never carries a resolved sha.
    - `raw_value` is the literal `'push'` (and the event cross-check above
      did not already refuse it): the push-to-main sentinel -- `None`
      (arm (b) intentionally skipped), with a printed notice so the skip
      shows up in the test's own stdout rather than disappearing.
    - `raw_value` is empty/`None` and `ci_mode` is true: a named
      `AssertionError` citing `JAMMI_CI_UNIT_HEAD_SHA` -- the matrix's own
      `|| 'push'` fallback means an empty value here can only be a wiring
      break (e.g. the matrix entry edited to drop the fallback), never a
      legitimate checkout of either event.
    - `raw_value` is empty/`None` and `ci_mode` is false: a bare local run
      with the var unset -- `None`, with a printed notice, exactly like
      the local-run doctrine this module's own top-of-file doc names.
    - anything else: resolved via `git rev-parse --verify <v>^{commit}` to
      its FULL sha (a full sha, an abbreviated sha, or a ref name all
      work identically) so arm (b) always compares full-sha-to-full-sha;
      a value `git` cannot resolve is a named `AssertionError` citing the
      malformed value itself -- NEVER folded into `STALE_ARTIFACT_RECORD_
      MESSAGE`, since an unresolvable env var is a wiring bug, not
      evidence the artifact is out of date.

    When `event_name` is anything else (including `None` -- a bare local
    run, where `GITHUB_EVENT_NAME` is normally unset), the two event
    cross-checks above never engage and only the plain `raw_value`/
    `ci_mode` cases apply, unchanged from before this function knew about
    `GITHUB_EVENT_NAME` at all.
    """
    if event_name in _PR_EVENT_NAMES and raw_value == _PUSH_SENTINEL:
        raise AssertionError(
            f"{_EVENT_NAME_ENV_VAR}={event_name!r} but {_UNIT_HEAD_ENV_VAR}={_PUSH_SENTINEL!r} -- a pull_request "
            f"checkout must export the PR's own head sha, never the push-to-main sentinel; check both "
            f"{_EVENT_NAME_ENV_VAR} and {_UNIT_HEAD_ENV_VAR}"
        )
    if event_name == _PUSH_SENTINEL and raw_value and raw_value != _PUSH_SENTINEL:
        raise AssertionError(
            f"{_EVENT_NAME_ENV_VAR}={_PUSH_SENTINEL!r} but {_UNIT_HEAD_ENV_VAR}={raw_value!r} -- a push-to-main "
            f"checkout must carry the literal sentinel {_PUSH_SENTINEL!r}, never a resolved sha; check both "
            f"{_EVENT_NAME_ENV_VAR} and {_UNIT_HEAD_ENV_VAR}"
        )
    if raw_value == _PUSH_SENTINEL:
        print(f"[frontend_ab_artifact] {_UNIT_HEAD_ENV_VAR}={_PUSH_SENTINEL!r} -- arm (b) intentionally skipped "
              "(push-to-main checkout, no PR head to compare against)")
        return None
    if not raw_value:
        if ci_mode:
            raise AssertionError(
                f"{_UNIT_HEAD_ENV_VAR} is empty/unset -- ci.yml's own matrix entry exports "
                f"`${{{{ github.event.pull_request.head.sha || '{_PUSH_SENTINEL}' }}}}`, so an empty value here "
                f"means that export itself broke, never a legitimate push-to-main checkout (which carries the "
                f"literal string {_PUSH_SENTINEL!r})"
            )
        print(f"[frontend_ab_artifact] {_UNIT_HEAD_ENV_VAR} unset -- arm (b) skipped (local run)")
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", f"{raw_value}^{{commit}}"],
        cwd=repo_root, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"{_UNIT_HEAD_ENV_VAR}={raw_value!r} does not resolve to a commit in {repo_root} "
            f"(`git rev-parse --verify {raw_value}^{{commit}}` failed: {proc.stderr.strip()}) -- refusing by "
            "name rather than diagnosing this as artifact staleness"
        )
    return proc.stdout.strip()


def assert_artifact_commit_is_unit_head(artifact_rel_path: str, repo_root: Path, unit_head_sha: str | None) -> None:
    """Arm (b): only when `unit_head_sha` (the unit's own PR head sha,
    e.g. CI's `JAMMI_CI_UNIT_HEAD_SHA`) is given, assert that `C`
    (`artifact_rel_path`'s own last commit) IS that head -- a no-op when
    `unit_head_sha` is `None`/empty (a push-to-main checkout, where arm (a)
    alone applies). Any later commit on the PR after `C` -- whether or not
    it touches a file arm (a)'s own file list already names -- fails this,
    since arm (a) alone has no way to see a commit that never touched the
    artifact file itself.
    """
    if not unit_head_sha:
        return
    c = _artifact_own_last_commit(artifact_rel_path, repo_root)
    if c != unit_head_sha:
        raise AssertionError(
            f"{STALE_ARTIFACT_RECORD_MESSAGE}: {artifact_rel_path}'s own last commit ({c}) is not this unit's "
            f"own PR head ({unit_head_sha!r}) -- a later commit landed on the PR after the artifact was last "
            "rendered"
        )


def assert_committed_artifact_not_stale(
    committed: dict,
    regenerated: dict,
    artifact_rel_path: str,
    repo_root: Path,
    unit_head_sha: str | None = None,
) -> None:
    """Runs arm (a) (`assert_committed_artifact_record_matches_artifact_
    commit`) always, then arm (b) (`assert_artifact_commit_is_unit_head`)
    whenever `unit_head_sha` is given -- see this module's own doc, "The
    'measured tip precedes merge tip' record is a FIXPOINT of `C`" section,
    for the full argument for why the two are independent checks.
    """
    assert_committed_artifact_record_matches_artifact_commit(committed, regenerated, artifact_rel_path, repo_root)
    assert_artifact_commit_is_unit_head(artifact_rel_path, repo_root, unit_head_sha)


def _init_repo(root: Path) -> tuple[str, str]:
    _git(["init", "-q"], root)
    (root / "unit").mkdir()
    (root / "unit" / "hot.rs").write_text("v0\n", encoding="utf-8")
    _git(["add", "."], root)
    _git(["commit", "-q", "-m", "base"], root)
    base_sha = _git(["rev-parse", "HEAD"], root).stdout.strip()

    (root / "unit" / "hot.rs").write_text("v1\n", encoding="utf-8")
    _git(["commit", "-q", "-am", "tip: touches the hot path"], root)
    tip_sha = _git(["rev-parse", "HEAD"], root).stdout.strip()
    return base_sha, tip_sha


def _write_raw_leg(
    raw_dir: Path, tower: str, role: str, repeat: str, *, build_sha: str,
    front_per_step: float, train_per_step: float, steps: int = STEPS,
    rayon_pool_threads: int | None = None, exit_code: int = 0, report_override: dict | None = None,
    task: str | None = None, device_name: str = DEVICE_NAME,
) -> None:
    stem = f"{tower}__{role}__{repeat}"
    (raw_dir / f"{stem}.exit").write_text(str(exit_code), encoding="utf-8")
    (raw_dir / f"{stem}.stderr").write_text("", encoding="utf-8")
    if report_override is not None:
        report = report_override
    else:
        tier = {
            "steps_measured": steps,
            "media_front_end_wall_s": front_per_step * steps,
            "train_run_wall_s": train_per_step * steps,
            "task": task if task is not None else TASK_BY_TOWER[tower],
            "device_name": device_name,
        }
        if rayon_pool_threads is not None:
            tier["rayon_pool_threads"] = rayon_pool_threads
        report = {
            "host": {"logical_cpus": rayon_pool_threads or 4},
            "provenance": {"build_sha": build_sha, "target": "x86_64-unknown-linux-gnu", "profile": "release"},
            "tiers": {"finetune_run": tier},
        }
    (raw_dir / f"{stem}.json").write_text(json.dumps(report), encoding="utf-8")


def _write_serial_tail(root: Path, *, audio_t_s: float | str | None = MEASURED_SERIAL_TAIL_S, extra_lines: list[str] | None = None) -> Path:
    """A synthetic stand-in for the committed `fixtures/frontend_ab_final/
    serial_tail.txt` -- the same 'task=... t_s=...' line shape, never the
    real file (that is exercised separately by `RealFixtureRegressionTests`
    below). `audio_t_s=None` omits the audio_embedding line entirely (for
    the 'missing line' refusal test); a string value writes it verbatim
    (for the 'malformed t_s' refusal test)."""
    lines = []
    if audio_t_s is not None:
        lines.append(f"task=audio_embedding device=cuda:0 reps=20 dims=[24, 4, 1001, 64] t_s={audio_t_s}")
    lines.append("task=image_embedding device=cuda:0 reps=20 dims=[24, 3, 224, 224] t_s=0.0011608887")
    if extra_lines:
        lines.extend(extra_lines)
    path = root / "serial_tail.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_fixture(root: Path, base_sha: str, tip_sha: str, *, repeats: tuple[str, ...] = REPEATS) -> dict:
    raw_dir = root / "raw"
    raw_dir.mkdir()
    for tower in TOWERS:
        for role in ROLES:
            sha = base_sha if role == "base" else tip_sha
            for repeat in repeats:
                _write_raw_leg(
                    raw_dir, tower, role, repeat, build_sha=sha,
                    front_per_step=FRONT_PER_STEP[(tower, role)],
                    train_per_step=TRAIN_PER_STEP[(tower, role)],
                    rayon_pool_threads=P if role == "tip" else None,
                )

    towers_json = {
        tower: {
            role: {
                repeat: {
                    "outcome": "OK",
                    "steps_measured": STEPS,
                    "front_per_step": FRONT_PER_STEP[(tower, role)],
                    "train_per_step": TRAIN_PER_STEP[(tower, role)],
                    "rayon_pool_threads": P if role == "tip" else None,
                }
                for repeat in repeats
            }
            for role in ROLES
        }
        for tower in TOWERS
    }
    htsat_bar = _oracle_bar(
        [FRONT_PER_STEP[("htsat", "base")]] * len(repeats),
        [FRONT_PER_STEP[("htsat", "tip")]] * len(repeats),
        P, N_ITEMS_PER_STEP, R_DRIVER,
    )
    cv_front_tip = FRONT_PER_STEP[("clip-vision", "tip")]
    cv_front_base = FRONT_PER_STEP[("clip-vision", "base")]
    report_json = {
        "tool": "frontend_ab.sh",
        "dry_run": False,
        "base_sha": base_sha,
        "tip_sha": tip_sha,
        "box": BOX,
        "serial_tail_ratio": R_DRIVER,
        "n_items_per_step": N_ITEMS_PER_STEP,
        "repeats": len(repeats),
        "status": "GREEN",
        "towers": towers_json,
        "htsat_bar": htsat_bar,
        "clip_vision_report_only": {
            "front_tip_mean_s": cv_front_tip,
            "front_base_mean_s": cv_front_base,
            "ratio": cv_front_tip / cv_front_base,
        },
    }
    identity = {
        "what": "fixture identity",
        "gpu": "FIXTURE-GPU",
        "driver": "0.0.0",
        "cpu": "FIXTURE-CPU",
        "recorded_deviations": ["fixture deviation"],
        "measured_tip_precedes_merge_tip_commentary": "fixture commentary",
    }
    report_path = root / "report.json"
    identity_path = root / "identity.json"
    report_path.write_text(json.dumps(report_json), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    serial_tail_path = _write_serial_tail(root)
    return {
        "raw_dir": raw_dir,
        "report_path": report_path,
        "report_json": report_json,
        "identity_path": identity_path,
        "identity": identity,
        "serial_tail_path": serial_tail_path,
        "htsat_bar_oracle": htsat_bar,
    }


class HappyPathTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, serial_tail_path=None):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"],
            serial_tail_path if serial_tail_path is not None else self.fixture["serial_tail_path"], self.root,
            "python3 ci/scripts/perf/frontend_ab_artifact.py (test invocation)",
        )

    def test_top_level_shape(self):
        report = self._build()
        self.assertEqual(set(report.keys()), {"schema_version", "git_sha", "box", "producer", "status", "notes", "measurement", "verdict"})
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["git_sha"], self.tip_sha)
        self.assertEqual(report["box"], "TESTBOX")
        self.assertEqual(report["producer"]["path"], "ci/scripts/perf/frontend_ab_artifact.py")
        self.assertEqual(report["producer"]["kind"], "script")
        self.assertEqual(report["producer"]["gating"], "none")
        self.assertEqual(report["status"], "GREEN")

    def test_htsat_bar_driver_matches_independent_oracle(self):
        report = self._build()
        oracle = _oracle_bar(
            [FRONT_PER_STEP[("htsat", "base")]] * 3, [FRONT_PER_STEP[("htsat", "tip")]] * 3, P, N_ITEMS_PER_STEP, R_DRIVER
        )
        got = report["measurement"]["htsat_bar_driver_r"]
        for field, expected in oracle.items():
            if isinstance(expected, float):
                self.assertAlmostEqual(got[field], expected, places=9, msg=field)
            else:
                self.assertEqual(got[field], expected, msg=field)
        self.assertEqual(oracle["verdict"], "PASS")  # sanity: this fixture is a PASS scenario

    def test_htsat_bar_measured_matches_independent_oracle(self):
        report = self._build()
        r_measured = MEASURED_SERIAL_TAIL_S / FRONT_PER_STEP[("htsat", "base")]
        oracle = _oracle_bar(
            [FRONT_PER_STEP[("htsat", "base")]] * 3, [FRONT_PER_STEP[("htsat", "tip")]] * 3, P, N_ITEMS_PER_STEP, r_measured
        )
        got = report["measurement"]["htsat_bar_measured_r"]
        for field, expected in oracle.items():
            if isinstance(expected, float):
                self.assertAlmostEqual(got[field], expected, places=9, msg=field)
            else:
                self.assertEqual(got[field], expected, msg=field)

    def test_verdict_block(self):
        report = self._build()
        v = report["verdict"]
        self.assertEqual(v["unit_verdict"], "PASS")
        self.assertEqual(v["cross_checked_against_report_json_htsat_bar_verdict"], "PASS")
        dev = v["serial_tail_ratio_deviation"]
        self.assertEqual(dev["kind"], "driver default vs measured; verdict invariant")
        self.assertAlmostEqual(dev["r_driver"], R_DRIVER, places=12)
        self.assertAlmostEqual(dev["r_measured"], MEASURED_SERIAL_TAIL_S / FRONT_PER_STEP[("htsat", "base")], places=9)
        self.assertEqual(dev["verdict_under_r_driver"], "PASS")
        self.assertEqual(dev["verdict_under_r_measured"], "PASS")
        self.assertTrue(dev["invariant_holds"])
        self.assertIn("ACTIVATE", v["contract_clause_applied"])

    def test_clip_vision_report_only_matches_oracle(self):
        report = self._build()
        cv = report["measurement"]["clip_vision_report_only"]
        expected_ratio = FRONT_PER_STEP[("clip-vision", "tip")] / FRONT_PER_STEP[("clip-vision", "base")]
        self.assertAlmostEqual(cv["ratio"], expected_ratio, places=9)
        self.assertAlmostEqual(cv["front_tip_mean_s"], FRONT_PER_STEP[("clip-vision", "tip")], places=9)
        self.assertAlmostEqual(cv["front_base_mean_s"], FRONT_PER_STEP[("clip-vision", "base")], places=9)

    def test_notes_carry_git_facts_and_hashes(self):
        report = self._build()
        notes = report["notes"]
        self.assertEqual(notes["driver_commit"], self.tip_sha)
        self.assertEqual(notes["rendered_from_tree_sha"], self.tip_sha)  # HEAD never moved past tip in this fixture
        self.assertIn("report_json", notes["input_sha256"])
        self.assertIn("identity", notes["input_sha256"])
        self.assertIn("serial_tail", notes["input_sha256"])
        # report+identity+serial_tail + a .json AND a .exit entry per raw leg
        self.assertEqual(notes["input_sha256"].keys() - {"report_json", "identity", "serial_tail"}, {
            f"raw/{tower}__{role}__{repeat}.{suffix}"
            for tower in TOWERS for role in ROLES for repeat in REPEATS for suffix in ("json", "exit")
        })
        self.assertEqual(len(notes["input_sha256"]), 3 + 2 * len(TOWERS) * len(ROLES) * len(REPEATS))
        self.assertTrue(all(len(v) == 64 for v in notes["input_sha256"].values()))
        self.assertEqual(len(notes["run_sha256"]), 64)
        self.assertIn("fixture deviation", notes["recorded_deviations"])
        dev = notes["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["kind"], "measured-tip-precedes-merge-tip")
        self.assertEqual(dev["measured_tip"], self.tip_sha)
        self.assertEqual(dev["rendered_from_tree_sha"], self.tip_sha)  # HEAD never moved past tip in this fixture
        self.assertEqual(dev["files_changed_since_measured_tip"], [])
        self.assertEqual(dev["commentary"], "fixture commentary")

    def test_exit_byte_divergence_moves_only_that_legs_hash_and_the_run_hash(self):
        """Advisory probe: a divergence confined to a
        `.exit` file's OWN bytes (`int(exit_text)` still `0` -- a valid
        leg, never `ArtifactBuildError`'s own `exit_code != 0` refusal)
        must move `input_sha256[that leg]` -- checked against an
        INDEPENDENT `hashlib.sha256` oracle, never just "some hash
        changed" -- and `run_sha256` (the run-wide digest folding every
        input in), and ONLY that leg's own `.exit` entry: no sibling leg's
        hash, and no OTHER input (`report_json`/`identity`/`serial_tail`)
        moves alongside it, so a divergence is traceable to the one leg
        that actually changed.
        """
        before = self._build()
        exit_path = self.fixture["raw_dir"] / "htsat__base__r1.exit"
        original_bytes = exit_path.read_bytes()
        self.assertEqual(original_bytes, b"0")  # exit_code=0, _write_raw_leg's own default
        exit_path.write_bytes(b"0\n")  # one-byte divergence; int("0\n") == 0, still a valid leg
        after = self._build()

        leg_key = "raw/htsat__base__r1.exit"
        self.assertEqual(before["notes"]["input_sha256"][leg_key], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(after["notes"]["input_sha256"][leg_key], hashlib.sha256(b"0\n").hexdigest())
        self.assertNotEqual(before["notes"]["input_sha256"][leg_key], after["notes"]["input_sha256"][leg_key])
        self.assertNotEqual(before["notes"]["run_sha256"], after["notes"]["run_sha256"])

        unaffected_keys = before["notes"]["input_sha256"].keys() - {leg_key}
        for key in unaffected_keys:
            self.assertEqual(
                before["notes"]["input_sha256"][key], after["notes"]["input_sha256"][key],
                msg=f"{key} moved from a divergence confined to {leg_key}",
            )

    def test_legs_present_for_every_tower_role_repeat(self):
        report = self._build()
        legs = report["measurement"]["legs"]
        self.assertEqual(len(legs), 4 * 3)
        for tower in TOWERS:
            for role in ROLES:
                for repeat in REPEATS:
                    key = f"{tower}__{role}__{repeat}"
                    self.assertIn(key, legs)
                    self.assertAlmostEqual(legs[key]["front_per_step"], FRONT_PER_STEP[(tower, role)], places=9)

    def test_regeneration_is_byte_identical(self):
        payload_a = json.dumps(self._build(), sort_keys=False)
        payload_b = json.dumps(self._build(), sort_keys=False)
        self.assertEqual(payload_a, payload_b)


class MeasuredTipPrecedesMergeTipTests(unittest.TestCase):
    """Files changed AFTER the measured tip are RECORDED mechanically
    (never refused: a branch tip that was "measured" can legitimately gain
    more commits before it merges -- exactly what happened on
    `perf/421-frontend` itself, a concurrent fold landing while this very
    artifact was being built), with the reviewed `--identity` commentary
    attached alongside the mechanical file list, never replacing it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"], self.fixture["serial_tail_path"], self.root, "inv",
        )

    def test_post_tip_change_is_recorded_mechanically_with_reviewed_commentary(self):
        (self.root / "docs").mkdir()
        (self.root / "docs" / "other.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "docs/other.md"], self.root)  # NEVER "add ." here -- the fixture's own raw_dir/report.json/
        # identity.json already sit untracked in this same tempdir (see _write_fixture) and would otherwise be
        # swept into this commit too.
        _git(["commit", "-q", "-m", "post-tip change"], self.root)
        head = _git(["rev-parse", "HEAD"], self.root).stdout.strip()

        report = self._build()
        self.assertEqual(report["notes"]["rendered_from_tree_sha"], head)
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["kind"], "measured-tip-precedes-merge-tip")
        self.assertEqual(dev["measured_tip"], self.tip_sha)
        self.assertEqual(dev["rendered_from_tree_sha"], head)
        self.assertEqual(dev["files_changed_since_measured_tip"], ["docs/other.md"])
        self.assertIn("docs/other.md", dev["diffstat"])
        self.assertEqual(dev["commentary"], "fixture commentary")

    def test_post_tip_change_that_touches_this_units_own_files_is_still_only_recorded(self):
        # A file-diff alone cannot judge whether an edit changed timed
        # behaviour -- even a change to the SAME file this unit's own scope
        # touched (`unit/hot.rs`) is recorded, never refused; that judgement
        # lives in --identity's own reviewed commentary, not in this
        # mechanism.
        (self.root / "unit" / "hot.rs").write_text("v2 -- touched again after the measured tip\n", encoding="utf-8")
        _git(["commit", "-q", "-am", "post-tip, touches the same file again"], self.root)

        report = self._build()
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["files_changed_since_measured_tip"], ["unit/hot.rs"])

    def test_no_post_tip_changes_is_recorded_as_an_empty_list(self):
        report = self._build()
        dev = report["notes"]["measured_tip_precedes_merge_tip"]
        self.assertEqual(dev["files_changed_since_measured_tip"], [])
        self.assertEqual(dev["diffstat"], "")

    def test_shallow_repo_is_refused(self):
        with mock.patch.object(art, "_is_shallow_repository", return_value=True):
            with self.assertRaises(art.ArtifactBuildError) as ctx:
                self._build()
        self.assertIn("shallow checkout", str(ctx.exception))


class ArtifactRecordFreshnessGateTests(unittest.TestCase):
    """Exercises `assert_committed_artifact_not_stale`'s two arms on a
    small scratch `git init`'d repo, independent of the real committed
    fixture -- `RealFixtureRegressionTests.
    test_committed_artifact_record_is_not_stale` below drives the SAME
    gate against the real repo.

    Six scenarios:
      (1) the artifact is regenerated and committed as the unit's OWN LAST
          commit, and the unit's own PR head sha (arm (b)) matches that
          commit -> green.
      (2) one MORE commit lands afterwards, touching a file that was
          ALREADY in the record's own file list, with the unit's own PR
          head sha set to that later commit -> arm (a) alone stays green
          (it never looks past `C`), but the combined gate (arm (b)
          engaged) is RED, named.
      (3) one MORE commit lands afterwards, touching a file that was NEVER
          in the record's own file list, with the unit's own PR head sha
          set to that later commit -> same result as (2): arm (a) green,
          combined gate RED.
      (4) no PR head sha is known (a push-to-main checkout) and a later,
          unrelated MERGE commit sits somewhere in history after `C` ->
          arm (a) alone stays green (it is a property of `C`/`C^`, never
          of the checkout's own HEAD).
      (5) same as (4), but the merge commit itself IS the checkout's
          current HEAD (a genuine two-parent commit) -> still green.
      (6) the committed record is tampered with (a file dropped from its
          own `files_changed_since_measured_tip`) with NO later commits at
          all -> RED under arm (a) alone.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(["init", "-q"], self.root)
        (self.root / "src.rs").write_text("v0\n", encoding="utf-8")
        _git(["add", "."], self.root)
        _git(["commit", "-q", "-m", "measured tip"], self.root)
        self.measured_tip = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        self.artifact_path = self.root / "artifact.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _render_and_commit_artifact(self, measurement: dict, message: str) -> dict:
        """Mirrors the real producer's own discipline (see
        `frontend_ab_artifact.py`'s "Regeneration discipline" section):
        `rendered_from_tree_sha` is `git rev-parse HEAD` taken BEFORE this
        commit is made -- the producer writes the file, and only THEN does
        a human/CI commit it."""
        rendered_from_tree_sha = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        files_changed = sorted(art._diff_name_only(self.measured_tip, rendered_from_tree_sha, self.root))
        diffstat = art._diff_stat(self.measured_tip, rendered_from_tree_sha, self.root).strip()
        report = {
            "measurement": measurement,
            "notes": {
                "rendered_from_tree_sha": rendered_from_tree_sha,
                "measured_tip_precedes_merge_tip": {
                    "kind": "measured-tip-precedes-merge-tip",
                    "measured_tip": self.measured_tip,
                    "rendered_from_tree_sha": rendered_from_tree_sha,
                    "files_changed_since_measured_tip": files_changed,
                    "diffstat": diffstat,
                    "commentary": "synthetic",
                },
            },
        }
        self.artifact_path.write_text(json.dumps(report), encoding="utf-8")
        _git(["add", "artifact.json"], self.root)
        _git(["commit", "-q", "-m", message], self.root)
        return report

    def _regenerate(self, measurement: dict) -> dict:
        """A fresh-at-HEAD regeneration -- real `git diff` output, never
        hand-typed, so this mirrors what `build_report` itself would
        compute at the CURRENT tip."""
        head = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        files_changed = sorted(art._diff_name_only(self.measured_tip, head, self.root))
        return {
            "measurement": measurement,
            "notes": {
                "rendered_from_tree_sha": head,
                "measured_tip_precedes_merge_tip": {
                    "kind": "measured-tip-precedes-merge-tip",
                    "measured_tip": self.measured_tip,
                    "rendered_from_tree_sha": head,
                    "files_changed_since_measured_tip": files_changed,
                    "diffstat": "synthetic-regenerated",
                    "commentary": "synthetic",
                },
            },
        }

    def test_scenario1_final_commit_with_matching_pr_head_is_green(self):
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        regenerated = self._regenerate({"m": 1})
        c = _artifact_own_last_commit("artifact.json", self.root)
        assert_committed_artifact_not_stale(
            committed, regenerated, "artifact.json", self.root, unit_head_sha=c
        )  # must not raise

    def test_scenario1b_second_legitimate_rerender_with_matching_pr_head_is_green(self):
        # A SECOND, legitimate re-render: only artifact.json changes again
        # (e.g. the fix for a scenario-(2)-shaped finding is "re-run the
        # producer", not "touch a source file").
        self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        committed = self._render_and_commit_artifact({"m": 1}, "re-render artifact only")
        regenerated = self._regenerate({"m": 1})
        c = _artifact_own_last_commit("artifact.json", self.root)
        assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root, unit_head_sha=c)

    def test_scenario2_later_commit_touching_an_already_listed_file_is_red_under_arm_b(self):
        # `src.rs` changes BEFORE the artifact is rendered, so it is already
        # part of the record's own files_changed_since_measured_tip --
        # then it changes AGAIN afterwards.
        (self.root / "src.rs").write_text("v1 -- touched before the render\n", encoding="utf-8")
        _git(["commit", "-q", "-am", "pre-render change"], self.root)
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        self.assertIn("src.rs", committed["notes"]["measured_tip_precedes_merge_tip"]["files_changed_since_measured_tip"])

        (self.root / "src.rs").write_text("v2 -- touched again after the artifact's own last commit\n", encoding="utf-8")
        _git(["commit", "-q", "-am", "post-artifact change to an already-listed file"], self.root)
        later_commit = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        regenerated = self._regenerate({"m": 1})

        # Arm (a) alone never looks past `C` -- still green.
        assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root, unit_head_sha=None)
        # Arm (b), engaged with the PR's own (later) head sha, catches it.
        with self.assertRaises(AssertionError) as ctx:
            assert_committed_artifact_not_stale(
                committed, regenerated, "artifact.json", self.root, unit_head_sha=later_commit
            )
        self.assertIn(STALE_ARTIFACT_RECORD_MESSAGE, str(ctx.exception))

    def test_scenario3_later_commit_touching_a_never_listed_file_is_red_under_arm_b(self):
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        (self.root / "docs").mkdir()
        (self.root / "docs" / "other.md").write_text("hello\n", encoding="utf-8")
        _git(["add", "docs/other.md"], self.root)  # NEVER "add ." here -- see _write_fixture's own comment above.
        _git(["commit", "-q", "-m", "post-artifact change to a brand-new, never-listed file"], self.root)
        later_commit = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        regenerated = self._regenerate({"m": 1})

        assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root, unit_head_sha=None)
        with self.assertRaises(AssertionError) as ctx:
            assert_committed_artifact_not_stale(
                committed, regenerated, "artifact.json", self.root, unit_head_sha=later_commit
            )
        self.assertIn(STALE_ARTIFACT_RECORD_MESSAGE, str(ctx.exception))

    def test_scenario4_no_pr_head_known_with_a_later_unrelated_merge_present_stays_green(self):
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], self.root).stdout.strip()
        _git(["checkout", "-q", "-b", "side"], self.root)
        (self.root / "side.txt").write_text("side change\n", encoding="utf-8")
        _git(["add", "side.txt"], self.root)
        _git(["commit", "-q", "-m", "unrelated side-branch commit"], self.root)
        _git(["checkout", "-q", default_branch], self.root)
        _git(["merge", "--no-ff", "-q", "-m", "merge side into the default branch", "side"], self.root)
        # One more commit AFTER the merge -- the merge commit is present in
        # history, but not itself HEAD (that is scenario (5) below).
        (self.root / "after_merge.txt").write_text("after the merge\n", encoding="utf-8")
        _git(["add", "after_merge.txt"], self.root)
        _git(["commit", "-q", "-m", "one more unrelated commit after the merge"], self.root)

        regenerated = self._regenerate({"m": 1})
        assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root, unit_head_sha=None)

    def test_scenario5_two_parent_merge_commit_as_head_with_no_pr_head_known_is_green(self):
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        default_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], self.root).stdout.strip()
        _git(["checkout", "-q", "-b", "side"], self.root)
        (self.root / "side.txt").write_text("side change\n", encoding="utf-8")
        _git(["add", "side.txt"], self.root)
        _git(["commit", "-q", "-m", "unrelated side-branch commit"], self.root)
        _git(["checkout", "-q", default_branch], self.root)
        _git(["merge", "--no-ff", "-q", "-m", "merge side into the default branch", "side"], self.root)
        head = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        parents = _git(["rev-list", "--parents", "-n", "1", head], self.root).stdout.strip().split()
        self.assertEqual(len(parents), 3, "HEAD must be a genuine two-parent merge commit")  # sha + 2 parents

        regenerated = self._regenerate({"m": 1})
        assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root, unit_head_sha=None)

    def test_scenario6_tampered_record_with_a_file_removed_is_red_under_arm_a(self):
        (self.root / "extra.rs").write_text("v0\n", encoding="utf-8")
        _git(["add", "extra.rs"], self.root)
        _git(["commit", "-q", "-m", "pre-render change touching a second file"], self.root)
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        files = committed["notes"]["measured_tip_precedes_merge_tip"]["files_changed_since_measured_tip"]
        self.assertIn("extra.rs", files)
        tampered = json.loads(json.dumps(committed))
        tampered["notes"]["measured_tip_precedes_merge_tip"]["files_changed_since_measured_tip"] = [
            f for f in files if f != "extra.rs"
        ]

        regenerated = self._regenerate({"m": 1})
        with self.assertRaises(AssertionError) as ctx:
            assert_committed_artifact_not_stale(tampered, regenerated, "artifact.json", self.root, unit_head_sha=None)
        self.assertIn(STALE_ARTIFACT_RECORD_MESSAGE, str(ctx.exception))
        self.assertIn("extra.rs", str(ctx.exception))

    def test_measurement_drift_is_a_named_failure_even_with_a_fresh_render(self):
        committed = self._render_and_commit_artifact({"m": 1}, "add artifact (final commit)")
        regenerated = self._regenerate({"m": 2})  # a number moved, nothing else did
        with self.assertRaises(AssertionError) as ctx:
            assert_committed_artifact_not_stale(committed, regenerated, "artifact.json", self.root)
        self.assertIn(STALE_ARTIFACT_RECORD_MESSAGE, str(ctx.exception))
        self.assertIn("measurement block is not byte-identical", str(ctx.exception))


class UnitHeadShaResolutionTests(unittest.TestCase):
    """Exercises `resolve_unit_head_sha` -- the normaliser standing between
    the raw `JAMMI_CI_UNIT_HEAD_SHA` env var and arm (b)
    (`assert_artifact_commit_is_unit_head`) -- on a small scratch `git
    init`'d repo, independent of both the real fixture
    (`RealFixtureRegressionTests`) and `ArtifactRecordFreshnessGateTests`'s
    own repo. Every case below either engages arm (b) with a real resolved
    sha, or explicitly records (via a printed notice, asserted present in
    no test here -- `stdout` is not captured by this harness, only the
    return value / raised exception are) that arm (b) was intentionally
    skipped, or REFUSES by name.

    Scratch-repo scenarios: PR-lane value == `C` -> green; PR-lane value
    != `C` -> RED (stale message); `push` -> arm (a) only, green; empty
    under `ci_mode=True` -> RED naming the variable; empty under
    `ci_mode=False` -> skipped, no RED; `not-a-sha` -> RED naming the
    malformed input (never the stale message); an abbreviated sha of `C`
    -> green (resolves to the full sha); and, event-keyed:
    `GITHUB_EVENT_NAME=pull_request` (or `pull_request_target`) with the
    `push` sentinel -> RED naming both `GITHUB_EVENT_NAME` and
    `JAMMI_CI_UNIT_HEAD_SHA`; `GITHUB_EVENT_NAME=push` with a resolved sha
    -> RED naming both variables as inconsistent; `GITHUB_EVENT_NAME=push`
    with the `push` sentinel -> green; no `GITHUB_EVENT_NAME` at all (a
    bare local run) -> the event cross-check never engages, so `push`
    still skips arm (b) regardless of `ci_mode`.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _git(["init", "-q"], self.root)
        (self.root / "artifact.json").write_text("{}\n", encoding="utf-8")
        _git(["add", "."], self.root)
        _git(["commit", "-q", "-m", "C: the artifact's own last commit"], self.root)
        self.c = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        self.c_short = self.c[:8]

    def tearDown(self):
        self._tmp.cleanup()

    def test_pr_lane_value_equal_to_c_is_green(self):
        resolved = resolve_unit_head_sha(self.c, self.root, ci_mode=True)
        self.assertEqual(resolved, self.c)
        assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)  # must not raise

    def test_pr_lane_value_not_equal_to_c_is_red_with_stale_message(self):
        (self.root / "other.txt").write_text("later\n", encoding="utf-8")
        _git(["add", "other.txt"], self.root)
        _git(["commit", "-q", "-m", "a later commit that never touches artifact.json"], self.root)
        later = _git(["rev-parse", "HEAD"], self.root).stdout.strip()
        resolved = resolve_unit_head_sha(later, self.root, ci_mode=True)
        self.assertEqual(resolved, later)
        with self.assertRaises(AssertionError) as ctx:
            assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)
        self.assertIn(STALE_ARTIFACT_RECORD_MESSAGE, str(ctx.exception))

    def test_push_sentinel_skips_arm_b_and_is_green_in_either_mode(self):
        for ci_mode in (True, False):
            with self.subTest(ci_mode=ci_mode):
                resolved = resolve_unit_head_sha("push", self.root, ci_mode=ci_mode)
                self.assertIsNone(resolved)
                assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)  # arm (a)-only, no-op

    def test_empty_value_under_ci_mode_is_red_naming_the_variable(self):
        for raw_value in ("", None):
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(AssertionError) as ctx:
                    resolve_unit_head_sha(raw_value, self.root, ci_mode=True)
                self.assertIn(_UNIT_HEAD_ENV_VAR, str(ctx.exception))

    def test_empty_value_outside_ci_mode_skips_arm_b_without_a_red(self):
        for raw_value in ("", None):
            with self.subTest(raw_value=raw_value):
                resolved = resolve_unit_head_sha(raw_value, self.root, ci_mode=False)
                self.assertIsNone(resolved)

    def test_malformed_value_is_red_naming_the_malformed_input_not_staleness(self):
        with self.assertRaises(AssertionError) as ctx:
            resolve_unit_head_sha("not-a-sha", self.root, ci_mode=True)
        message = str(ctx.exception)
        self.assertIn("not-a-sha", message)
        self.assertNotIn(STALE_ARTIFACT_RECORD_MESSAGE, message)

    def test_pull_request_event_with_pr_head_sha_is_green(self):
        resolved = resolve_unit_head_sha(self.c, self.root, ci_mode=True, event_name="pull_request")
        self.assertEqual(resolved, self.c)
        assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)  # must not raise

    def test_pull_request_event_with_push_sentinel_is_red_naming_both_variables(self):
        with self.assertRaises(AssertionError) as ctx:
            resolve_unit_head_sha("push", self.root, ci_mode=True, event_name="pull_request")
        message = str(ctx.exception)
        self.assertIn(_EVENT_NAME_ENV_VAR, message)
        self.assertIn(_UNIT_HEAD_ENV_VAR, message)

    def test_pull_request_target_event_with_push_sentinel_is_also_red(self):
        with self.assertRaises(AssertionError) as ctx:
            resolve_unit_head_sha("push", self.root, ci_mode=True, event_name="pull_request_target")
        message = str(ctx.exception)
        self.assertIn(_EVENT_NAME_ENV_VAR, message)
        self.assertIn(_UNIT_HEAD_ENV_VAR, message)

    def test_push_event_with_push_sentinel_is_green(self):
        resolved = resolve_unit_head_sha("push", self.root, ci_mode=True, event_name="push")
        self.assertIsNone(resolved)
        assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)  # arm (a)-only, no-op

    def test_push_event_with_a_resolved_sha_is_red_naming_both_variables_as_inconsistent(self):
        with self.assertRaises(AssertionError) as ctx:
            resolve_unit_head_sha(self.c, self.root, ci_mode=True, event_name="push")
        message = str(ctx.exception)
        self.assertIn(_EVENT_NAME_ENV_VAR, message)
        self.assertIn(_UNIT_HEAD_ENV_VAR, message)

    def test_no_event_name_keeps_current_behaviour_push_sentinel_still_skips(self):
        for ci_mode in (True, False):
            with self.subTest(ci_mode=ci_mode):
                resolved = resolve_unit_head_sha("push", self.root, ci_mode=ci_mode, event_name=None)
                self.assertIsNone(resolved)

    def test_abbreviated_sha_of_c_resolves_to_full_sha_and_is_green(self):
        resolved = resolve_unit_head_sha(self.c_short, self.root, ci_mode=True)
        self.assertEqual(resolved, self.c)
        self.assertNotEqual(resolved, self.c_short)  # actually normalised, not just accepted verbatim
        assert_artifact_commit_is_unit_head("artifact.json", self.root, resolved)  # must not raise


class RefusalTests(unittest.TestCase):
    """Every way the input can be missing, failed, mismatched, or non-finite
    must refuse -- never silently pick a value or propagate a NaN."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture, serial_tail_path=None):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"],
            serial_tail_path if serial_tail_path is not None else fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_missing_exit_file(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__base__r1.exit").unlink()
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("missing .exit file", str(ctx.exception))

    def test_refuses_nonzero_exit(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__tip__r2.exit").write_text("3", encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("leg exited 3", str(ctx.exception))

    def test_refuses_build_sha_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r1.json").read_text())
        bad["provenance"]["build_sha"] = "f" * 40
        (fixture["raw_dir"] / "htsat__base__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not match the base sha", str(ctx.exception))

    def test_refuses_nonfinite_media_front_end_wall_s(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "clip-vision__tip__r1.json").read_text())
        bad["tiers"]["finetune_run"]["media_front_end_wall_s"] = math.nan
        (fixture["raw_dir"] / "clip-vision__tip__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_infinite_train_run_wall_s(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r3.json").read_text())
        bad["tiers"]["finetune_run"]["train_run_wall_s"] = math.inf
        (fixture["raw_dir"] / "htsat__base__r3.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("not finite", str(ctx.exception))

    def test_refuses_front_per_step_disagreement_with_report_json(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["towers"]["htsat"]["tip"]["r1"]["front_per_step"] = 999.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("front_per_step", str(ctx.exception))

    def test_refuses_htsat_bar_verdict_disagreement(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["htsat_bar"]["verdict"] = "FAIL"
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagrees with --report-json", str(ctx.exception))

    def test_refuses_rayon_pool_threads_disagreement_across_tip_legs(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__tip__r2.json").read_text())
        bad["tiers"]["finetune_run"]["rayon_pool_threads"] = 999
        (fixture["raw_dir"] / "htsat__tip__r2.json").write_text(json.dumps(bad), encoding="utf-8")
        fixture["report_json"]["towers"]["htsat"]["tip"]["r2"]["rayon_pool_threads"] = 999
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("disagree on rayon_pool_threads", str(ctx.exception))

    def test_refuses_base_leg_with_unexpected_rayon_pool_threads(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad = json.loads((fixture["raw_dir"] / "htsat__base__r1.json").read_text())
        bad["tiers"]["finetune_run"]["rayon_pool_threads"] = 8
        (fixture["raw_dir"] / "htsat__base__r1.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("unexpectedly carries rayon_pool_threads", str(ctx.exception))

    def test_refuses_verdict_invariant_violation(self):
        # A measured serial tail large enough to push r_measured's own
        # verdict off PASS while r_driver=0.0 stays PASS -- the "verdict
        # invariant" this contract names must be a checked equality, not
        # merely asserted prose.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad_serial_tail = _write_serial_tail(self.root, audio_t_s=0.5)  # r_measured = 0.5 -> way past PASS
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture, serial_tail_path=bad_serial_tail)
        self.assertIn("verdict invariant", str(ctx.exception))

    def test_refuses_missing_towers_key(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        del fixture["report_json"]["towers"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("missing required key 'towers'", str(ctx.exception))

    def test_refuses_leg_outcome_not_ok(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["towers"]["clip-vision"]["base"]["r1"]["outcome"] = "FAIL"
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("own outcome is", str(ctx.exception))


class LegSetEnumerationTests(unittest.TestCase):
    """`_validate_leg_set`/`_validate_towers_repeat_keys` -- the leg set on
    `--raw-dir` and `--report-json`'s own `towers` object must equal
    EXACTLY `TOWERS x ROLES x {r1..r<repeats>}`, in BOTH directions. This
    covers the audit's own probe: `--repeats` shrinking from 3 to 2 while
    the `r3` raw-leg files are STILL PRESENT on disk (e.g. a stale re-run
    left them behind) must REFUSE, never silently read only r1/r2 and
    PASS."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"], fixture["serial_tail_path"], self.root, "inv",
        )

    def test_extra_leg_on_disk_beyond_declared_repeats_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha, repeats=("r1", "r2", "r3"))
        fixture["report_json"]["repeats"] = 2
        for tower in TOWERS:
            for role in ROLES:
                del fixture["report_json"]["towers"][tower][role]["r3"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not carry exactly", str(ctx.exception))
        self.assertIn("extra leg stems", str(ctx.exception))

    def test_missing_leg_on_disk_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "htsat__tip__r3.json").unlink()
        (fixture["raw_dir"] / "htsat__tip__r3.exit").unlink()
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("does not carry exactly", str(ctx.exception))
        self.assertIn("missing leg stems", str(ctx.exception))

    def test_stray_extra_file_of_an_unknown_tower_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        (fixture["raw_dir"] / "unknown-tower__base__r1.json").write_text("{}", encoding="utf-8")
        (fixture["raw_dir"] / "unknown-tower__base__r1.exit").write_text("0", encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("extra leg stems", str(ctx.exception))

    def test_report_json_towers_with_extra_repeat_key_is_refused(self):
        # --raw-dir itself carries EXACTLY the repeats=2 leg set, but
        # --report-json's own towers.htsat.tip object still carries a
        # leftover 'r3' entry -- independent of whatever sits on disk.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha, repeats=("r1", "r2"))
        fixture["report_json"]["towers"]["htsat"]["tip"]["r3"] = dict(
            fixture["report_json"]["towers"]["htsat"]["tip"]["r1"]
        )
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("towers.htsat.tip", str(ctx.exception))

    def test_report_json_towers_missing_a_repeat_key_is_refused(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        del fixture["report_json"]["towers"]["clip-vision"]["base"]["r2"]
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("towers.clip-vision.base", str(ctx.exception))


class TowerTaskAndDeviceNameTests(unittest.TestCase):
    """Advisory cross-check: every leg's own `tiers.finetune_run.task`/
    `.device_name` must match the tower being read / `--report-json`'s own
    `box` device prefix -- refusing by name on either mismatch, rather than
    trusting the filename alone."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"], fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_task_tower_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "htsat__base__r1.json"
        report = json.loads(raw_path.read_text())
        report["tiers"]["finetune_run"]["task"] = "image_embedding"
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("tiers.finetune_run.task", str(ctx.exception))

    def test_refuses_device_name_mismatch(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "clip-vision__tip__r2.json"
        report = json.loads(raw_path.read_text())
        report["tiers"]["finetune_run"]["device_name"] = "SOME OTHER GPU"
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("device_name", str(ctx.exception))

    def test_refuses_missing_task_field(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "htsat__tip__r3.json"
        report = json.loads(raw_path.read_text())
        del report["tiers"]["finetune_run"]["task"]
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("carries no tiers.finetune_run.task", str(ctx.exception))

    def test_refuses_missing_device_name_field(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        raw_path = fixture["raw_dir"] / "clip-vision__base__r1.json"
        report = json.loads(raw_path.read_text())
        del report["tiers"]["finetune_run"]["device_name"]
        raw_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("carries no tiers.finetune_run.device_name", str(ctx.exception))


class SerialTailFixtureTests(unittest.TestCase):
    """`_read_measured_serial_tail_s` -- named refusal if `--serial-tail`
    cannot be read, carries no `task=audio_embedding ... t_s=...` line, or
    that line's own `t_s` does not parse as a float."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, serial_tail_path):
        return art.build_report(
            self.fixture["raw_dir"], self.fixture["report_path"], self.fixture["report_json"],
            self.fixture["identity_path"], self.fixture["identity"], serial_tail_path, self.root, "inv",
        )

    def test_refuses_missing_serial_tail_file(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(self.root / "does-not-exist.txt")
        self.assertIn("could not read --serial-tail", str(ctx.exception))

    def test_refuses_missing_audio_task_line(self):
        bad = _write_serial_tail(self.root, audio_t_s=None)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(bad)
        self.assertIn("no 'task=audio_embedding ... t_s=...' line found", str(ctx.exception))

    def test_refuses_malformed_t_s(self):
        bad = _write_serial_tail(self.root, audio_t_s="not-a-float")
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(bad)
        self.assertIn("does not parse as a float", str(ctx.exception))


class RDomainTests(unittest.TestCase):
    """`r` (a serial-tail/front-end time ratio) is only defined on
    `[0, 1)` -- both edges, and beyond, are named refusals. Boundary
    matrix: r=0 (valid), r just under 1 (valid), r=1 (refused), r=1.5
    (refused), r=-0.5 (refused). Also: any zero/non-positive front-end
    time used as a divisor (`front_base_mean_s`==0) is a named
    `ArtifactBuildError`, never a `ZeroDivisionError`."""

    def test_validate_r_accepts_zero(self):
        self.assertEqual(art._validate_r(0.0, "r"), 0.0)

    def test_validate_r_accepts_just_under_one(self):
        self.assertAlmostEqual(art._validate_r(0.999999, "r"), 0.999999)

    def test_validate_r_refuses_exactly_one(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art._validate_r(1.0, "r")
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_validate_r_refuses_above_one(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._validate_r(1.5, "r")

    def test_validate_r_refuses_negative(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._validate_r(-0.5, "r")

    def test_require_positive_divisor_refuses_zero(self):
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            art._require_positive_divisor(0.0, "front_base")
        self.assertIn("must be a finite positive number to divide by", str(ctx.exception))

    def test_require_positive_divisor_refuses_negative(self):
        with self.assertRaises(art.ArtifactBuildError):
            art._require_positive_divisor(-1.0, "front_base")

    def test_require_positive_divisor_accepts_positive(self):
        self.assertEqual(art._require_positive_divisor(0.5, "front_base"), 0.5)


class RDomainEndToEndTests(unittest.TestCase):
    """The same boundary matrix as `RDomainTests`, driven end to end
    through `build_report` -- proving the domain guard is actually wired
    into the pipeline, not just a helper nothing calls."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, fixture, serial_tail_path=None):
        return art.build_report(
            fixture["raw_dir"], fixture["report_path"], fixture["report_json"],
            fixture["identity_path"], fixture["identity"],
            serial_tail_path if serial_tail_path is not None else fixture["serial_tail_path"], self.root, "inv",
        )

    def test_refuses_r_driver_equal_to_one(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = 1.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_driver_above_one(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = 1.5
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_driver_negative(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        fixture["report_json"]["serial_tail_ratio"] = -0.5
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_refuses_r_measured_above_one(self):
        # front_base_mean_s == 1.0 in this fixture's own HTSAT base legs,
        # so a measured t_s of 1.5 -> r_measured = 1.5.
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        bad_serial_tail = _write_serial_tail(self.root, audio_t_s=1.5)
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture, serial_tail_path=bad_serial_tail)
        self.assertIn("must be in [0.0, 1.0)", str(ctx.exception))

    def test_zero_htsat_front_base_is_a_named_refusal_not_zero_division(self):
        fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)
        for repeat in REPEATS:
            raw_path = fixture["raw_dir"] / f"htsat__base__{repeat}.json"
            report = json.loads(raw_path.read_text())
            report["tiers"]["finetune_run"]["media_front_end_wall_s"] = 0.0
            raw_path.write_text(json.dumps(report), encoding="utf-8")
            fixture["report_json"]["towers"]["htsat"]["base"][repeat]["front_per_step"] = 0.0
        with self.assertRaises(art.ArtifactBuildError) as ctx:
            self._build(fixture)
        self.assertIn("must be a finite positive number to divide by", str(ctx.exception))


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_sha, self.tip_sha = _init_repo(self.root)
        self.fixture = _write_fixture(self.root, self.base_sha, self.tip_sha)

    def tearDown(self):
        self._tmp.cleanup()

    def _run_cli(self, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable, str(ARTIFACT),
                "--raw-dir", str(self.fixture["raw_dir"]),
                "--report-json", str(self.fixture["report_path"]),
                "--identity", str(self.fixture["identity_path"]),
                "--serial-tail", str(self.fixture["serial_tail_path"]),
                "--repo-root", str(self.root),
                *extra_args,
            ],
            capture_output=True, text=True, timeout=120,
        )

    def test_cli_writes_a_schema_shaped_artifact(self):
        out_path = self.root / "artifact.json"
        result = self._run_cli("--out", str(out_path))
        self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
        report = json.loads(out_path.read_text())
        self.assertEqual(report["git_sha"], self.tip_sha)
        self.assertEqual(report["verdict"]["unit_verdict"], "PASS")
        self.assertIn("frontend_ab_artifact: git_sha=", result.stderr)

    def test_cli_refuses_and_exits_nonzero_on_bad_input(self):
        (self.fixture["raw_dir"] / "htsat__tip__r1.exit").write_text("7", encoding="utf-8")
        result = self._run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("leg exited 7", result.stderr)


def _find_committed_frontend_artifact(repo_root: Path, tip_sha: str) -> Path | None:
    """The ONE committed `crates/jammi-kernels/artifacts/cuda-runs/
    *-frontend-<8-char-short-sha>-*.json` artifact for this fixture's own
    `tip_sha` (the producer's own filename convention, e.g. `2026-09-08-
    frontend-0a8562c4-a100-pcie.json`) -- `None` if the directory is
    absent or zero/more-than-one file matches (an ambiguous match is a
    repo-layout finding for the caller to report, never silently picked
    from)."""
    cuda_runs_dir = repo_root / "crates" / "jammi-kernels" / "artifacts" / "cuda-runs"
    if not cuda_runs_dir.is_dir():
        return None
    short_sha = tip_sha[:8]
    matches = sorted(p for p in cuda_runs_dir.glob("*-frontend-*.json") if short_sha in p.name)
    if len(matches) != 1:
        return None
    return matches[0]


class RealFixtureRegressionTests(unittest.TestCase):
    """Drives the REAL, committed pod-p421c fixture (`fixtures/
    frontend_ab_final/`) through `build_report`, against the ACTUAL current
    checkout as `--repo-root` -- a regression guard that the committed
    `report.json` + raw legs still cross-check cleanly against this module's
    own re-derivation.

    Locally (or on any OTHER CI leg), this SKIPS -- never fails -- if the
    real fixture directory is absent, or if `--repo-root`'s own git history
    does not contain the fixture's `tip_sha`/`base_sha` (a shallow checkout
    of an unrelated repo running this file in isolation is not this test's
    concern). But `ci.yml`'s own `frontend_ab_artifact suite` matrix entry
    sets `JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED=1` (this repo's own
    zero-execution-is-RED doctrine, the same precedent
    `test_convert_legacy_bert_checkpoint.py::CiExecutionAssertionTests`
    already establishes for `JAMMI_CI_SAFETENSORS_EXPECTED`) AND
    `fetch_depth: "0"` -- under that env var, BOTH escape hatches turn into
    a hard `fail()` instead of a silent skip, so a regression that deletes
    the fixture directory, or a matrix-entry edit that drops `fetch_depth:
    "0"` and reintroduces a shallow checkout, shows up as a RED leg rather
    than an indistinguishable quiet skip."""

    _CI_ENV_VAR = "JAMMI_CI_FRONTEND_AB_FINAL_FIXTURE_EXPECTED"

    def _ci_expects_real_fixture(self) -> bool:
        return os.environ.get(self._CI_ENV_VAR) == "1"

    def _skip_or_fail(self, message: str) -> None:
        if self._ci_expects_real_fixture():
            self.fail(f"{message} -- but {self._CI_ENV_VAR}=1 (ci.yml's own matrix entry expects this to run)")
        self.skipTest(message)

    def test_real_fixture_cross_checks_cleanly(self):
        if not REAL_FIXTURE_DIR.is_dir():
            self._skip_or_fail(f"{REAL_FIXTURE_DIR} not present")
            return
        report_path = REAL_FIXTURE_DIR / "report.json"
        identity_path = PERF_DIR / "frontend_ab_final_identity.json"
        raw_dir = REAL_FIXTURE_DIR / "raw"
        serial_tail_path = REAL_FIXTURE_DIR / "serial_tail.txt"
        report_json = json.loads(report_path.read_text())
        identity = json.loads(identity_path.read_text())
        repo_root = PERF_DIR.parents[2]

        def _has_commit(sha: str) -> bool:
            proc = subprocess.run(["git", "cat-file", "-e", sha], cwd=repo_root, capture_output=True, text=True)
            return proc.returncode == 0

        if not (_has_commit(report_json["base_sha"]) and _has_commit(report_json["tip_sha"])):
            self._skip_or_fail(
                "this checkout's history does not contain the real fixture's base_sha/tip_sha "
                "(a shallow checkout -- fetch_depth: '0' is required for this matrix entry)"
            )
            return

        report = art.build_report(
            raw_dir, report_path, report_json, identity_path, identity,
            serial_tail_path, repo_root, "python3 ci/scripts/perf/frontend_ab_artifact.py (regression test)",
        )
        self.assertEqual(report["git_sha"], report_json["tip_sha"])
        self.assertEqual(report["verdict"]["unit_verdict"], "UNRESOLVED")
        self.assertEqual(report["measurement"]["htsat_bar_driver_r"]["verdict"], "UNRESOLVED")

    def test_committed_artifact_record_is_not_stale(self):
        """Arms (a) and (b) of `assert_committed_artifact_not_stale` (see
        this module's own doc, "The 'measured tip precedes merge tip'
        record is a FIXPOINT of `C`" section), driven against the REAL
        committed pod-p421c artifact.

        Arm (a) is HEAD-independent by construction and is expected GREEN
        at any tip: the committed record is checked directly against `C`
        (the artifact file's own last commit) and `C`'s own parent via git
        plumbing, never against whatever this checkout's own `HEAD`
        happens to be. Arm (b) additionally engages once `resolve_unit_
        head_sha` (see its own docstring) turns `JAMMI_CI_UNIT_HEAD_SHA`
        into a resolved sha -- `ci.yml`'s own matrix entry exports `${{
        github.event.pull_request.head.sha || 'push' }}`, so the raw value
        is the PR's own head sha on a `pull_request`/`pull_request_target`
        checkout, the literal `push` on a `push` checkout (arm (b)
        intentionally skipped), or -- outside CI, when this suite is run by
        hand with the var unset -- empty (also skipped, never a hard
        failure). `resolve_unit_head_sha` cross-checks the raw value
        against `GITHUB_EVENT_NAME` itself: the `push` sentinel on a
        `pull_request`/`pull_request_target` event, or any non-sentinel
        value on a `push` event, is REFUSED naming both variables, never
        silently resolved either way. Once engaged, arm (b) is green when
        the resolved sha equals `C`, RED (named) for any other value, since
        a later, un-rendered commit landed on the unit after the artifact
        was last regenerated; an unresolvable (malformed) value is a
        separate, differently-named RED that never masquerades as
        staleness.

        The committed pod-p421c artifact was regenerated at this unit's own
        final tip with the `.exit`-file fold already in place (see
        `frontend_ab_artifact.py`'s own hash-computation section), so
        `notes.input_sha256`/`notes.run_sha256` compare byte-identical to a
        fresh regeneration with no tolerance needed.
        """
        if not REAL_FIXTURE_DIR.is_dir():
            self._skip_or_fail(f"{REAL_FIXTURE_DIR} not present")
            return
        report_path = REAL_FIXTURE_DIR / "report.json"
        identity_path = PERF_DIR / "frontend_ab_final_identity.json"
        raw_dir = REAL_FIXTURE_DIR / "raw"
        serial_tail_path = REAL_FIXTURE_DIR / "serial_tail.txt"
        report_json = json.loads(report_path.read_text())
        identity = json.loads(identity_path.read_text())
        repo_root = PERF_DIR.parents[2]

        def _has_commit(sha: str) -> bool:
            proc = subprocess.run(["git", "cat-file", "-e", sha], cwd=repo_root, capture_output=True, text=True)
            return proc.returncode == 0

        if not (_has_commit(report_json["base_sha"]) and _has_commit(report_json["tip_sha"])):
            self._skip_or_fail(
                "this checkout's history does not contain the real fixture's base_sha/tip_sha "
                "(a shallow checkout -- fetch_depth: '0' is required for this matrix entry)"
            )
            return

        committed_path = _find_committed_frontend_artifact(repo_root, report_json["tip_sha"])
        if committed_path is None:
            self._skip_or_fail(
                "no single crates/jammi-kernels/artifacts/cuda-runs/*-frontend-*.json artifact for this "
                f"fixture's own tip_sha ({report_json['tip_sha']})"
            )
            return

        committed = json.loads(committed_path.read_text())
        # Regenerated with the COMMITTED record's own recorded invocation --
        # so `producer.invocation` is byte-identical between the two and
        # this gate never flags a divergence that is only "this test's
        # harness described itself differently", the same reason the
        # committed inputs (raw legs / report.json / identity / serial-tail)
        # are read from the SAME fixture files the committed artifact itself
        # was rendered from.
        regenerated = art.build_report(
            raw_dir, report_path, report_json, identity_path, identity,
            serial_tail_path, repo_root, committed["producer"]["invocation"],
        )
        artifact_rel_path = str(committed_path.relative_to(repo_root))
        unit_head_sha = resolve_unit_head_sha(
            os.environ.get(_UNIT_HEAD_ENV_VAR), repo_root, ci_mode=self._ci_expects_real_fixture(),
            event_name=os.environ.get(_EVENT_NAME_ENV_VAR),
        )
        assert_committed_artifact_not_stale(
            committed, regenerated, artifact_rel_path, repo_root, unit_head_sha=unit_head_sha
        )


if __name__ == "__main__":
    unittest.main()
