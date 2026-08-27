#!/usr/bin/env python3
"""Pod-build-timings artifact schema + provenance gate — hermetic, static,
no build, no GPU, no live pod.

## Why this exists

`ci/scripts/perf/pod_build_timings.sh` (the A2 producer, `docs/maintainer/
pod-build-guide.md` §6) writes a JSON artifact under
`ci/artifacts/pod-build-timings/<ts>-<sha7>.json` that `dev-gpu.md`'s own
`RP_DISK_GB` formula and `pod-build-guide.md`'s §4 time-budget both cite
numbers from directly. Nothing checked those numbers were shaped like real
measurements before this gate: a hand-edited or stale artifact could carry a
negative wall-clock, an unresolvable `git_sha`, or a `byte_equal_state`
outside its documented four-value vocabulary, and every citing doc would
still read as if it quoted a real, reproducible run. This is the accepted-
debt item the pod-substrate-followups audit named — modeled on
`check_cuda_run_artifacts.py`'s shape (schema typing, shallow-checkout guard,
ancestry-before-anything-else), scoped small: one artifact family, one
directory, no producer-static-verification rule (that family already has
its own `#[test]`/script-provenance machinery; this one has none yet).

## The schema (see the committed artifact + docs/maintainer/pod-build-guide.md §6)

Every `*.json` directly under `ci/artifacts/pod-build-timings/` must carry:

  - `schema_version` (int) — must be one of `KNOWN_SCHEMA_VERSIONS`.
  - `box` (non-empty string).
  - `git_sha` (40-hex, lowercase).
  - `ts` (ISO-8601 UTC, `YYYY-MM-DDTHH:MM:SSZ`, and must actually parse as a
    real calendar date/time — not merely shaped right).
  - `lock_held` — must be the boolean `true` (not merely truthy): the
    producer script's own timing-lock contract (`pod_timing_lock.sh`) is
    what makes a wall-clock number here trustworthy at all; a run that did
    NOT hold the lock is contaminated by construction (pod-build-guide.md
    §6's own "Contamination" note) and has no business being committed.
  - `measurements` (object), carrying:
      - the WALL fields (`clone_build_wall_s`, `cold_build_wall_s`,
        `flash_attn_leg_wall_s`, `copy_wall_s`) and BYTE fields
        (`S_src_bytes`, `S_seed_bytes`, `S_clone_bytes`): each a
        non-negative int, OR — ONLY `flash_attn_leg_wall_s`, and ONLY when
        `fa2_ran` is `false` with a non-empty `fa2_reason` alongside it (the
        "documented" half of "documented-null") — JSON `null`. Any other
        null is a hard FAIL: an undocumented null reads as a real
        measurement to anything that later cites it.
      - `fa2_ran` (bool) and `fa2_reason` (non-empty string).
  - `seed_tuples` (a list).
  - `byte_equal_state` — one of the four values pod-build-guide.md §6
    documents: `"invalid"`, `"set_mismatch"`, `"true"`, `"false"` (STRINGS,
    never a bare JSON boolean — the guide is explicit that this is
    deliberately never collapsed into one).

## Fail-closed contract

  (a) Every required field above is present and well-typed; findings are
      NAMED (which field, which file), never a bare non-zero exit.
  (b) `git_sha` must be an ancestor of `HEAD` (`git merge-base
      --is-ancestor`) — the same #406-merge-commit discipline
      `check_cuda_run_artifacts.py` rule (d) already enforces for cuda-run
      artifacts, and for the identical reason: a green artifact whose sha is
      not reachable from this branch is evidence about a tree that no longer
      exists. Checked BEFORE any ancestry work: a shallow checkout
      (`actions/checkout`'s default `fetch-depth: 1`) makes every single
      `git_sha` read back as a false non-ancestor, indistinguishable from a
      genuine one without this guard — one explicit failure naming the
      shallow checkout, never N misleading per-file findings.

Run: `python3 ci/scripts/check_pod_build_timings.py`
Self-test (RED cases for every rule above — every required field's own
typing, both halves of the null/documented-null split, the four-value
byte_equal_state vocabulary, a top-level-non-object payload, a JSON parse
error, an unreadable/non-UTF-8 file, ancestry against a REAL non-ancestor
commit plus a merge-reachable-only positive control, and the shallow-
checkout guard — on a throwaway `git init`'d fixture repo with an actual
branch/merge history, never the real checkout):
`python3 ci/scripts/check_pod_build_timings.py --self-test`
Hermetic: reads the working tree (or an ephemeral tempdir git repo under
`--self-test`) and shells out only to `git`; no network, no cargo, no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMINGS_DIR = REPO_ROOT / "ci" / "artifacts" / "pod-build-timings"

GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
KNOWN_SCHEMA_VERSIONS = {1}
BYTE_EQUAL_STATES = {"invalid", "set_mismatch", "true", "false"}

WALL_FIELDS = ("clone_build_wall_s", "cold_build_wall_s", "flash_attn_leg_wall_s", "copy_wall_s")
BYTE_FIELDS = ("S_src_bytes", "S_seed_bytes", "S_clone_bytes")
# Only this ONE field may legitimately be null, and only when documented by
# fa2_ran == false alongside a non-empty fa2_reason (see check_measurements).
NULLABLE_IF_FA2_SKIPPED = "flash_attn_leg_wall_s"

SHALLOW_CHECKOUT_MESSAGE = "shallow checkout — ancestry cannot be evaluated; use fetch-depth: 0"
ANCESTOR_MESSAGE = (
    "is not an ancestor of HEAD — a green artifact whose sha is not an ancestor of the "
    "branch is evidence about a tree that no longer exists (the #406 merge-commit discipline)."
)


class ArtifactError(Exception):
    """Uncomputable input (parse failure, missing dir) — fails closed."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def is_shallow_repository(repo_root: Path) -> bool:
    proc = _run(["git", "rev-parse", "--is-shallow-repository"], repo_root)
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _is_ancestor(sha: str, repo_root: Path, target: str = "HEAD") -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", sha, target], repo_root)
    return proc.returncode == 0


def _valid_ts(ts: str) -> bool:
    if not TS_RE.match(ts):
        return False
    try:
        # Python's fromisoformat wants +00:00, not a bare 'Z', pre-3.11.
        from datetime import datetime

        datetime.fromisoformat(ts[:-1] + "+00:00")
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# schema/typing
# --------------------------------------------------------------------------- #
def check_schema_types(data: dict) -> list[str]:
    failures: list[str] = []

    sv = data.get("schema_version")
    if not isinstance(sv, int) or isinstance(sv, bool):
        failures.append(f"schema_version must be an int, got {sv!r}")
    elif sv not in KNOWN_SCHEMA_VERSIONS:
        failures.append(f"schema_version {sv!r} is not a known version ({sorted(KNOWN_SCHEMA_VERSIONS)})")

    box = data.get("box")
    if not isinstance(box, str) or not box.strip():
        failures.append(f"box must be a non-empty string, got {box!r}")

    sha = data.get("git_sha")
    if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
        failures.append(f"git_sha must be 40 lowercase hex chars, got {sha!r}")

    ts = data.get("ts")
    if not isinstance(ts, str) or not _valid_ts(ts):
        failures.append(f"ts must be an ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ), got {ts!r}")

    lock_held = data.get("lock_held")
    if lock_held is not True:
        failures.append(f"lock_held must be the boolean true (a run that did not hold the lock is contaminated), got {lock_held!r}")

    seed_tuples = data.get("seed_tuples")
    if not isinstance(seed_tuples, list):
        failures.append(f"seed_tuples must be a list, got {seed_tuples!r}")

    byte_equal_state = data.get("byte_equal_state")
    if byte_equal_state not in BYTE_EQUAL_STATES:
        failures.append(
            f"byte_equal_state must be one of {sorted(BYTE_EQUAL_STATES)} (a string, never a bare "
            f"boolean), got {byte_equal_state!r}"
        )

    measurements = data.get("measurements")
    if not isinstance(measurements, dict):
        failures.append(f"measurements must be an object, got {measurements!r}")
    else:
        failures += check_measurements(measurements)

    return failures


def check_measurements(measurements: dict) -> list[str]:
    failures: list[str] = []

    fa2_ran = measurements.get("fa2_ran")
    if not isinstance(fa2_ran, bool):
        failures.append(f"measurements.fa2_ran must be a bool, got {fa2_ran!r}")

    fa2_reason = measurements.get("fa2_reason")
    if not isinstance(fa2_reason, str) or not fa2_reason.strip():
        failures.append(f"measurements.fa2_reason must be a non-empty string, got {fa2_reason!r}")

    documented_null_ok = fa2_ran is False and isinstance(fa2_reason, str) and fa2_reason.strip()

    for field in WALL_FIELDS + BYTE_FIELDS:
        if field not in measurements:
            failures.append(f"measurements missing required field `{field}`")
            continue
        v = measurements[field]
        if v is None:
            if field == NULLABLE_IF_FA2_SKIPPED and documented_null_ok:
                continue  # documented null: fa2 did not run, and it says why
            failures.append(
                f"measurements.{field} is null but is not a DOCUMENTED null (only "
                f"`{NULLABLE_IF_FA2_SKIPPED}` may be null, and only when fa2_ran is false with a "
                "non-empty fa2_reason) — an undocumented null reads as a real measurement to anything "
                "that later cites it"
            )
            continue
        if not isinstance(v, int) or isinstance(v, bool):
            failures.append(f"measurements.{field} must be a non-negative int (or a documented null), got {v!r}")
        elif v < 0:
            failures.append(f"measurements.{field} must be non-negative, got {v!r}")

    return failures


# --------------------------------------------------------------------------- #
# ancestry
# --------------------------------------------------------------------------- #
def check_ancestry(data: dict, repo_root: Path) -> list[str]:
    sha = data.get("git_sha")
    if not isinstance(sha, str) or not GIT_SHA_RE.match(sha):
        return []  # rule (a) already reported the malformed-sha failure
    if _is_ancestor(sha, repo_root):
        return []
    return [f"git_sha {sha} {ANCESTOR_MESSAGE}"]


# --------------------------------------------------------------------------- #
# gate driver
# --------------------------------------------------------------------------- #
def validate_artifact(data: dict, repo_root: Path) -> list[str]:
    failures = check_schema_types(data)
    failures += check_ancestry(data, repo_root)
    return failures


def run_gate(timings_dir: Path, repo_root: Path) -> list[str]:
    if not timings_dir.is_dir():
        raise ArtifactError(f"pod-build-timings dir not found: {timings_dir}")

    if is_shallow_repository(repo_root):
        raise ArtifactError(SHALLOW_CHECKOUT_MESSAGE)

    files = sorted(timings_dir.glob("*.json"))
    if not files:
        raise ArtifactError(f"no *.json artifacts found under {timings_dir}")

    all_failures: list[str] = []
    for f in files:
        relpath = f.relative_to(timings_dir).as_posix()
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # A NAMED finding, never an uncaught traceback: a file that
            # cannot even be READ (permissions, a binary/non-UTF-8 byte
            # sequence) is exactly as fail-closed as a JSON parse error
            # below — this gate's whole job is to never let an uncomputable
            # artifact read as "nothing to report".
            all_failures.append(f"{relpath}: could not read file: {e}")
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            all_failures.append(f"{relpath}: JSON parse error: {e}")
            continue
        if not isinstance(data, dict):
            all_failures.append(f"{relpath}: top-level JSON value is not an object")
            continue
        failures = validate_artifact(data, repo_root)
        all_failures.extend(f"{relpath}: {msg}" for msg in failures)

    return all_failures


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    try:
        failures = run_gate(TIMINGS_DIR, REPO_ROOT)
    except ArtifactError as exc:
        print(f"pod-build-timings: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    if failures:
        print("pod-build-timings: FAIL", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        print(f"\npod-build-timings: {len(failures)} finding(s).", file=sys.stderr)
        return 1

    print(
        f"pod-build-timings: PASS — every *.json under "
        f"{TIMINGS_DIR.relative_to(REPO_ROOT)} satisfies the schema and ancestry contract."
    )
    return 0


# --------------------------------------------------------------------------- #
# self-test — an ephemeral `git init`'d fixture repo, never the real
# checkout, proving each rule above actually bites.
# --------------------------------------------------------------------------- #
GOOD_ARTIFACT: dict = {
    "schema_version": 1,
    "box": "a100-sxm4 pod fixture (A100-SXM4-80GB)",
    # filled in with the fixture repo's own real commit sha at self-test time
    "git_sha": "",
    "ts": "2026-08-27T18:39:28Z",
    "lock_held": True,
    "measurements": {
        "clone_build_wall_s": 69,
        "cold_build_wall_s": 243,
        "flash_attn_leg_wall_s": 122,
        "fa2_ran": True,
        "fa2_reason": "declared and built",
        "copy_wall_s": 2,
        "S_src_bytes": 3552665600,
        "S_seed_bytes": 7775764480,
        "S_clone_bytes": 8149614592,
    },
    "seed_tuples": ["T1", "T2", "T3", "T1b"],
    "byte_equal_state": "false",
}


def _write_ancestry_fixture_repo(tmp: Path) -> dict[str, str]:
    """A richer fixture than a single linear commit, so the ancestry rule's
    own RED/GREEN legs exercise REAL git objects rather than a fabricated
    sha (advisory: a nonexistent 40-hex sha proves only "unknown object",
    never "a real commit that genuinely is not an ancestor"). Builds:

        c1 (root) -- c2 (main) -------- m1 (HEAD, merge commit)
          \\                            /
           side (s1, NEVER merged)   feature (f1)

    `m1` (HEAD) has two parents: `c2` and `f1`. `s1` is a real, valid
    commit object that is NOT reachable from `m1` at all (a genuine
    non-ancestor, never merged into anything) — the RED case. `f1` is
    reachable from `m1` ONLY through the merge's second parent, never
    through a linear walk of `main` alone — the merge-commit-reachable-only
    POSITIVE control, proving `git merge-base --is-ancestor` (and therefore
    this gate) correctly credits ancestry through a merge, not just a
    straight line.
    """
    # `-b main`: pins the initial branch name explicitly rather than relying
    # on the host's own `init.defaultBranch` (which is "main" on some
    # installs, "master" on older/unconfigured ones) — this fixture's later
    # `git checkout -q main` needs the name to be deterministic.
    _run(["git", "init", "-q", "-b", "main"], tmp)
    _run(["git", "config", "user.email", "test@example.com"], tmp)
    _run(["git", "config", "user.name", "Test"], tmp)
    (tmp / "README.md").write_text("root\n", encoding="utf-8")
    _run(["git", "add", "README.md"], tmp)
    _run(["git", "commit", "-q", "-m", "c1 (root)"], tmp)

    _run(["git", "checkout", "-q", "-b", "side"], tmp)
    (tmp / "side.txt").write_text("side\n", encoding="utf-8")
    _run(["git", "add", "side.txt"], tmp)
    _run(["git", "commit", "-q", "-m", "s1 (side, never merged)"], tmp)
    side_sha = _run(["git", "rev-parse", "HEAD"], tmp).stdout.strip()

    _run(["git", "checkout", "-q", "main"], tmp)
    (tmp / "main2.txt").write_text("main2\n", encoding="utf-8")
    _run(["git", "add", "main2.txt"], tmp)
    _run(["git", "commit", "-q", "-m", "c2 (main)"], tmp)

    _run(["git", "checkout", "-q", "-b", "feature"], tmp)
    (tmp / "feature.txt").write_text("feature\n", encoding="utf-8")
    _run(["git", "add", "feature.txt"], tmp)
    _run(["git", "commit", "-q", "-m", "f1 (feature)"], tmp)
    feature_sha = _run(["git", "rev-parse", "HEAD"], tmp).stdout.strip()

    _run(["git", "checkout", "-q", "main"], tmp)
    _run(["git", "merge", "-q", "--no-ff", "-m", "m1 (merge feature into main)", "feature"], tmp)
    head_sha = _run(["git", "rev-parse", "HEAD"], tmp).stdout.strip()

    return {"head": head_sha, "non_ancestor": side_sha, "merge_reachable_only": feature_sha}


def self_test() -> int:
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td)
        shas = _write_ancestry_fixture_repo(repo_root)
        timings_dir = repo_root / "ci" / "artifacts" / "pod-build-timings"
        timings_dir.mkdir(parents=True)

        def write(name: str, data: dict) -> None:
            (timings_dir / name).write_text(json.dumps(data), encoding="utf-8")

        def good() -> dict:
            d = json.loads(json.dumps(GOOD_ARTIFACT))
            d["git_sha"] = shas["head"]
            return d

        # --- control: the good fixture itself must pass clean -------------
        write("good.json", good())
        got = run_gate(timings_dir, repo_root)
        if got:
            failures.append(f"self-test FAILED: control (good fixture) reported findings: {got}")
        (timings_dir / "good.json").unlink()

        # --- missing field --------------------------------------------------
        d = good()
        del d["box"]
        write("missing-field.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("box" in g for g in got):
            failures.append(f"self-test FAILED: missing `box` field not caught: {got}")
        (timings_dir / "missing-field.json").unlink()

        # --- top-level JSON value is not an object --------------------------
        (timings_dir / "not-an-object.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        got = run_gate(timings_dir, repo_root)
        if not any("not-an-object.json" in g and "not an object" in g for g in got):
            failures.append(f"self-test FAILED: a top-level JSON array (not an object) not caught: {got}")
        (timings_dir / "not-an-object.json").unlink()

        # --- JSON parse error -------------------------------------------------
        (timings_dir / "bad-json.json").write_text("{not valid json", encoding="utf-8")
        got = run_gate(timings_dir, repo_root)
        if not any("bad-json.json" in g and "JSON parse error" in g for g in got):
            failures.append(f"self-test FAILED: a JSON parse error not caught: {got}")
        (timings_dir / "bad-json.json").unlink()

        # --- could not read file (invalid UTF-8) — a NAMED finding, never an
        # uncaught traceback (advisory) --------------------------------------
        (timings_dir / "bad-encoding.json").write_bytes(b"\xff\xfe\x00not utf-8")
        got = run_gate(timings_dir, repo_root)
        if not any("bad-encoding.json" in g and "could not read file" in g for g in got):
            failures.append(f"self-test FAILED: an unreadable/non-UTF-8 file did not produce a named finding: {got}")
        (timings_dir / "bad-encoding.json").unlink()

        # --- bad git_sha (malformed) ----------------------------------------
        d = good()
        d["git_sha"] = "not-a-real-sha"
        write("bad-sha-shape.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("git_sha" in g and "40 lowercase hex" in g for g in got):
            failures.append(f"self-test FAILED: malformed git_sha not caught: {got}")
        (timings_dir / "bad-sha-shape.json").unlink()

        # --- bad git_sha (a REAL commit that genuinely is not an ancestor) ---
        # A fabricated 40-hex sha (the OLD form of this leg) only proves
        # "unknown object" — indistinguishable from a typo. `non_ancestor`
        # is a real, valid commit object in THIS fixture repo that was
        # deliberately never merged into anything, so this proves the rule
        # actually walks history, not merely that `git` rejects a sha it has
        # never heard of.
        d = good()
        d["git_sha"] = shas["non_ancestor"]
        write("bad-sha-ancestry.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("not an ancestor of HEAD" in g for g in got):
            failures.append(f"self-test FAILED: a REAL non-ancestor git_sha not caught: {got}")
        (timings_dir / "bad-sha-ancestry.json").unlink()

        # --- positive control: a sha reachable ONLY through a merge --------
        # `merge_reachable_only` is not on `main`'s own first-parent line at
        # all — it is `feature`'s tip, reachable from HEAD only via the
        # merge commit's SECOND parent. Must PASS, proving ancestry credit
        # flows through a merge and this gate is not silently linear-only.
        d = good()
        d["git_sha"] = shas["merge_reachable_only"]
        write("merge-reachable-control.json", d)
        got = run_gate(timings_dir, repo_root)
        if got:
            failures.append(
                f"self-test FAILED: a sha reachable ONLY through a merge commit was flagged as a "
                f"non-ancestor (ancestry-through-merge is broken): {got}"
            )
        (timings_dir / "merge-reachable-control.json").unlink()

        # --- unknown schema_version (well-typed int, not in KNOWN_SCHEMA_VERSIONS) --
        d = good()
        d["schema_version"] = 999
        write("unknown-schema-version.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("schema_version" in g and "not a known version" in g for g in got):
            failures.append(f"self-test FAILED: an unknown schema_version not caught: {got}")
        (timings_dir / "unknown-schema-version.json").unlink()

        # --- seed_tuples not a list -------------------------------------------
        d = good()
        d["seed_tuples"] = "T1,T2"
        write("seed-tuples-not-list.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("seed_tuples" in g and "must be a list" in g for g in got):
            failures.append(f"self-test FAILED: seed_tuples not a list not caught: {got}")
        (timings_dir / "seed-tuples-not-list.json").unlink()

        # --- measurements not an object ---------------------------------------
        d = good()
        d["measurements"] = "not an object"
        write("measurements-not-object.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("measurements" in g and "must be an object" in g for g in got):
            failures.append(f"self-test FAILED: measurements not an object not caught: {got}")
        (timings_dir / "measurements-not-object.json").unlink()

        # --- fa2_ran / fa2_reason typing ---------------------------------------
        d = good()
        d["measurements"]["fa2_ran"] = "yes"  # must be a bool
        d["measurements"]["fa2_reason"] = ""  # must be non-empty
        write("fa2-typing.json", d)
        got = run_gate(timings_dir, repo_root)
        if not (
            any("fa2_ran" in g and "must be a bool" in g for g in got)
            and any("fa2_reason" in g and "non-empty" in g for g in got)
        ):
            failures.append(f"self-test FAILED: fa2_ran/fa2_reason typing not caught: {got}")
        (timings_dir / "fa2-typing.json").unlink()

        # --- a bool where an int is required (bool IS an int in Python) --------
        d = good()
        d["measurements"]["copy_wall_s"] = True
        write("wall-field-is-bool.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("copy_wall_s" in g and "non-negative int" in g for g in got):
            failures.append(f"self-test FAILED: a bool value on a wall field (isinstance(True, int) is True in Python) not caught: {got}")
        (timings_dir / "wall-field-is-bool.json").unlink()

        # --- negative wall value ---------------------------------------------
        d = good()
        d["measurements"]["clone_build_wall_s"] = -1
        write("negative-wall.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("clone_build_wall_s" in g and "non-negative" in g for g in got):
            failures.append(f"self-test FAILED: negative wall value not caught: {got}")
        (timings_dir / "negative-wall.json").unlink()

        # --- undocumented null -------------------------------------------------
        d = good()
        d["measurements"]["cold_build_wall_s"] = None
        write("undocumented-null.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("cold_build_wall_s" in g and "DOCUMENTED null" in g for g in got):
            failures.append(f"self-test FAILED: undocumented null not caught: {got}")
        (timings_dir / "undocumented-null.json").unlink()

        # --- documented null (control: must NOT be flagged) --------------------
        d = good()
        d["measurements"]["flash_attn_leg_wall_s"] = None
        d["measurements"]["fa2_ran"] = False
        d["measurements"]["fa2_reason"] = "not on main; JAMMI_MAIN_SHA not set"
        write("documented-null.json", d)
        got = run_gate(timings_dir, repo_root)
        if got:
            failures.append(f"self-test FAILED: a legitimately documented null (fa2_ran=false + reason) was flagged: {got}")
        (timings_dir / "documented-null.json").unlink()

        # --- bad byte_equal_state -----------------------------------------------
        d = good()
        d["byte_equal_state"] = True  # a bare boolean, never valid
        write("bad-byte-equal.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("byte_equal_state" in g for g in got):
            failures.append(f"self-test FAILED: a bare-boolean byte_equal_state not caught: {got}")
        (timings_dir / "bad-byte-equal.json").unlink()

        # --- lock_held not true -----------------------------------------------
        d = good()
        d["lock_held"] = False
        write("lock-not-held.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("lock_held" in g for g in got):
            failures.append(f"self-test FAILED: lock_held=false not caught: {got}")
        (timings_dir / "lock-not-held.json").unlink()

        # --- bad ts shape -----------------------------------------------------
        d = good()
        d["ts"] = "not-a-timestamp"
        write("bad-ts.json", d)
        got = run_gate(timings_dir, repo_root)
        if not any("ts must be" in g for g in got):
            failures.append(f"self-test FAILED: malformed ts not caught: {got}")
        (timings_dir / "bad-ts.json").unlink()

        # --- shallow-checkout guard --------------------------------------------
        # Committed (not just written) so a `--depth 1` clone of this fixture
        # repo actually carries the file at all -- an untracked file is
        # invisible to `git clone` regardless of depth, which would make the
        # clone below fail on "dir not found" rather than the shallow-guard
        # message this leg exists to prove.
        write("good2.json", good())
        _run(["git", "add", "ci/artifacts/pod-build-timings/good2.json"], repo_root)
        _run(["git", "commit", "-q", "-m", "shallow-guard fixture"], repo_root)
        with tempfile.TemporaryDirectory() as td2:
            clone_dir = Path(td2) / "shallow-clone"
            # `file://` is REQUIRED to force a genuine shallow clone: a bare
            # local filesystem path silently IGNORES --depth ("--depth is
            # ignored in local clones; use file:// instead"), which would
            # make this fixture indistinguishable from a normal clone.
            clone_proc = _run(
                ["git", "clone", "-q", "--depth", "1", "file://" + str(repo_root), str(clone_dir)], Path(td2)
            )
            if clone_proc.returncode != 0:
                failures.append(f"self-test FAILED: could not create a --depth 1 clone fixture: {clone_proc.stderr}")
            else:
                shallow_timings_dir = clone_dir / "ci" / "artifacts" / "pod-build-timings"
                try:
                    run_gate(shallow_timings_dir, clone_dir)
                    failures.append("self-test FAILED: run_gate did not raise on a shallow checkout")
                except ArtifactError as exc:
                    if str(exc) != SHALLOW_CHECKOUT_MESSAGE:
                        failures.append(f"self-test FAILED: shallow-checkout ArtifactError had the wrong message: {exc}")
        (timings_dir / "good2.json").unlink()

    if failures:
        print("pod-build-timings self-test: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(
        "pod-build-timings self-test: OK — every rule bites: schema_version (incl. an unknown-but-"
        "well-typed version), box/git_sha-shape/ts-shape, lock_held, seed_tuples-not-a-list, "
        "measurements-not-an-object, fa2_ran/fa2_reason typing, a bool on a wall field, "
        "documented/undocumented null, byte_equal_state vocabulary, top-level-not-an-object, a JSON "
        "parse error, an unreadable/non-UTF-8 file (a named finding, never a traceback), ancestry "
        "against a REAL non-ancestor commit plus a merge-reachable-only positive control, and the "
        "shallow-checkout guard."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
