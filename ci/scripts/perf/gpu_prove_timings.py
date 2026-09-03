#!/usr/bin/env python3
"""GPU-prove-lane timing/outcome artifact producer (esc-080/esc-083).

Parses a `gpu-prove.yml` job log (the raw, tab-
separated `<job>\\t<step>\\t<ISO8601 timestamp> <message>` form `gh run view
--log` emits) and writes one `ci/artifacts/gpu-prove-timings/<run_id>-
<arch>.json` artifact — the evidence `ci/scripts/check_gpu_prove_timings.py`
re-checks on every run (R1-R5: the two watchdog defaults are actually set,
`RP_INACTIVITY` covers the worst observed silent gap with margin,
`RP_TIMEOUT`'s two-term backstop holds against every healthy wall, every
cut/kill carries a disposition, and every shipped arch has a fresh
same-surface healthy artifact).

Two modes:

  * CURRENT (default) -- the post-esc-081 script shape: `::group::<token>`/
    `::endgroup::` pairs use the SIX renamed short tokens plus `device`/
    `bench`, each proof group's own `PROVE_GROUP_RC name=<n> rc=<v>` marker,
    one `PROVE_SHA=<sha>` echo after clone, `PROVE_TUPLE crate=<c> kind=<k>
    features=<literal>` echoes, and a final `PROVE_EXIT=<n>` line (present
    only on an in-suite exit, per the 124/76 discriminator). `surface.
    expected_id` is computed from the log's own `PROVE_TUPLE` lines via
    `ci/scripts/prove_surface.py`'s shared canonicalization -- NOT the
    current tree's manifest, so a committed artifact's claimed surface never
    silently drifts if this script is re-run later against a newer checkout.
  * LEGACY (`--legacy`) -- a pre-esc-081 job log: descriptive `::group::`
    titles (mapped to the six canonical group names via prefix match, see
    `LEGACY_TITLE_PREFIXES`), no `PROVE_GROUP_RC`/`PROVE_SHA`/`PROVE_TUPLE`
    markers at all. `run_id`/`job_id`/`git_sha`/`outcome` cannot be derived
    from the log body (GitHub run metadata, external to the log) and MUST be
    passed on the command line; `source` is recorded as `"run-metadata"`,
    every `groups[].rc` is `null` (no per-group marker existed), and
    `surface.expected_id` is `null` (a legacy leg's surface predates this
    canonicalization entirely -- `check_gpu_prove_timings.py`'s R5 only
    demands a matching `expected_id` from `prove-lane`-kind artifacts, never
    `legacy-pre-d1` ones, precisely so these seeds never trip it).

No `cargo metadata`, no network -- pure stdlib text parsing plus
`ci/scripts/prove_surface.py` (itself `tomllib`-only).

Usage:
  python3 ci/scripts/perf/gpu_prove_timings.py --arch sm_80 --run-id <r> \\
      --job-id <j> [--box <name>] [--driver <ver>] [--git-sha <sha>] \\
      [--outcome <o>] [--out <path>] <job-log>

  python3 ci/scripts/perf/gpu_prove_timings.py --legacy --arch sm_86 \\
      --run-id <r> --job-id <j> --git-sha <sha> --outcome budget-cut \\
      [--out <path>] <job-log>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import prove_surface  # noqa: E402

SCHEMA_VERSION = 1
ARTIFACT_DIR = REPO_ROOT / "ci" / "artifacts" / "gpu-prove-timings"

# The six gating groups (matches `PROVE_GROUPS` in runpod_gpu_prove.sh) plus
# the two non-gating groups this producer still records timing for.
CURRENT_GROUP_NAMES = frozenset(
    {
        "device",
        "capability-surface-build",
        "capability-surface-proof",
        "served-client-server-proof",
        "engine-core-sweep",
        "kernels-default",
        "kernels-cuda",
        "bench",
    }
)

# Legacy (pre-esc-081) descriptive `::group::` titles -> canonical short
# name, matched by PREFIX (a legacy title carries trailing parenthetical
# prose the canonical name drops). `jammi-kernels clippy` has NO current
# equivalent (esc-081 removed it from the prove lane entirely) -- kept under
# its own informational name so a legacy leg's silence/wall accounting still
# sees it, but it is never compared against `CURRENT_GROUP_NAMES`.
LEGACY_TITLE_PREFIXES: list[tuple[str, str]] = [
    ("device", "device"),
    ("capability-surface build", "capability-surface-build"),
    ("capability-surface proof", "capability-surface-proof"),
    ("served client/server GPU proof", "served-client-server-proof"),
    ("engine-core GPU correctness", "engine-core-sweep"),
    ("GPU embedding perf", "bench"),
    ("jammi-kernels lib tests, default features", "kernels-default"),
    ("jammi-kernels lib tests, --features cuda", "kernels-cuda"),
    ("jammi-kernels clippy", "kernels-clippy-legacy"),
]

_LOG_LINE_RE = re.compile(r"^(?P<job>[^\t]*)\t(?P<step>[^\t]*)\t(?P<ts>\S+)\s(?P<msg>.*)$")
_GROUP_RE = re.compile(r"##\[group\](?P<title>.*)$")
_ENDGROUP_RE = re.compile(r"##\[endgroup\]\s*$")
# The ONE Python-side marker grammar (BLOCK 3 audit fix) -- imported from
# prove_surface.py rather than compiled here a second time, so this and the
# bash-side `rp_parse_prove_marker` (runpod_lib.sh) cannot silently drift.
_GROUP_RC_RE = prove_surface.PROVE_GROUP_RC_RE
_PROVE_SHA_RE = prove_surface.PROVE_SHA_RE
# esc-084/#454: the driver's own `WRONG TREE` diagnostic
# (`runpod_lib.sh`'s `rp_run_remote_watched`) names the sha it expected and
# the sha it actually saw (or the literal token `none` when no `PROVE_SHA=`
# line was ever observed by session end). Checked against the WHOLE log
# text (like `_DRIVER_EXIT_LINE_RE`'s BUDGET/NO PROGRESS siblings below),
# never per-line inside the measured window. BLOCK B9 audit fix: the
# diagnostic does NOT only fire before any proof group opens or after the
# window has closed with nothing else in the log -- the driver's final-
# flush and absence arms can (and do) leave a `PROVE_EXIT=0` line and every
# gating group green in the SAME log as this diagnostic, when the remote
# script itself ran to completion believing it succeeded and the driver
# only caught the wrong tree in its own final flush. `build_artifact` below
# therefore tests THIS match before `has_prove_exit`, not after -- the
# driver's 77 wins regardless of which other markers are present.
_WRONG_TREE_RE = re.compile(r"WRONG TREE expected=(?P<expected>\S+) got=(?P<got>\S+)")
_PROVE_TUPLE_RE = re.compile(r"PROVE_TUPLE crate=(?P<crate>\S+) kind=(?P<kind>\S+) features=(?P<features>\S*)")
_PROVE_EXIT_RE = re.compile(r"PROVE_EXIT=(?P<rc>-?\d+)")
_DEVICE_CSV_RE = re.compile(r"^(?P<name>[^,]+),\s*[\d.]+,\s*(?P<driver>[\d.]+)\s*$")


def _parse_ts(ts: str) -> datetime:
    ts = ts.lstrip("﻿")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _iter_log_lines(text: str):
    for raw in text.splitlines():
        m = _LOG_LINE_RE.match(raw)
        if not m:
            continue
        try:
            ts = _parse_ts(m.group("ts"))
        except ValueError:
            continue
        yield ts, m.group("msg")


def _legacy_group_name(title: str) -> str | None:
    for prefix, canon in LEGACY_TITLE_PREFIXES:
        if title.startswith(prefix):
            return canon
    return None


_DRIVER_EXIT_LINE_RE = re.compile(r"GPU prove suites exit=")


def _canon_group_name(title: str, legacy: bool) -> str | None:
    """The one place CURRENT and LEGACY modes agree on what counts as a
    prove-lane group at all -- everything else in a raw job log (the
    GitHub runner's own step-boundary `##[group]Run actions/checkout@v4`/
    `##[group]Operating System`/etc. groups, sitting BEFORE this script's
    own first group) is not a group this producer has any business timing.
    LEGACY already filtered through `_legacy_group_name`'s prefix map;
    CURRENT gets the same treatment against the exact `CURRENT_GROUP_NAMES`
    set instead of accepting any title verbatim (the D5 measurement-scope
    bug: an unfiltered CURRENT-mode title let the runner's own provisioning
    groups anchor the window)."""
    if legacy:
        return _legacy_group_name(title)
    return title if title in CURRENT_GROUP_NAMES else None


def parse_log(text: str, legacy: bool) -> dict:
    """Returns {groups: [{name, wall_s, rc}], max_silent_gap_s,
    max_silent_gap_after, git_sha (or None), tuples: {(crate,kind): feat},
    prove_exit (or None), box, driver, wall_s}.

    The WALL/silence window is bounded to [first CANONICAL `::group::` open
    (`device`, or the first of the canonical tokens to appear if `device`
    is absent), last of {a CANONICAL group's own `::endgroup::`, a
    `PROVE_EXIT=` line, a "GPU prove suites exit=" driver line}] -- NOT the
    whole job log. A raw job log also carries the GitHub runner's own
    provisioning/checkout step groups (several minutes) BEFORE this
    script's first `::group::device` line, and the runner's own post-job
    cleanup step groups AFTER the driver's own exit echo; a naive "first/
    last `##[group]`/`##[endgroup]` in the file" anchor silently absorbs
    both into `wall_s`/`max_silent_gap_s` (the D5 measurement-scope bug --
    confirmed live: sm_90's reported `max_silent_gap_s` was the pod-
    provisioning wait, sm_80's reported `wall_s` was the WHOLE job's
    duration). Neither runner phase belongs to the prove lane's own wall or
    silence accounting; a non-canonical `##[group]` title occurring OUTSIDE
    this window is ignored entirely, and one occurring INSIDE it (the
    driver holds exactly one canonical group open at a time in this range,
    by construction) is a parse error, not a silently-dropped group.
    """
    events = list(_iter_log_lines(text))
    if not events:
        raise ValueError("no parseable timestamped lines in this log")

    open_groups: dict[str, datetime] = {}
    closed: list[dict] = []
    current_open_name: str | None = None
    # BLOCK 1 audit fix: `PROVE_GROUP_RC` markers are emitted BEFORE their
    # own `::endgroup::` by design (runpod_gpu_prove.sh's own convention),
    # so at the moment a marker is seen its group is still in `open_groups`,
    # never yet in `closed` -- searching `closed` for a name match (the
    # original shape) found nothing for EVERY current-mode marker, silently
    # leaving `rc: None` on every group and reading a fully healthy log as
    # `budget-cut`. Attribute by the marker's OWN `name=` field into a plain
    # dict instead (the marker already states which group it belongs to;
    # only one group is ever open at a time, so this needs no additional
    # "is it the currently-open one" check) and pull from that dict the
    # moment the SAME-named group closes.
    rc_by_name: dict[str, int] = {}
    git_sha: str | None = None
    tuples: dict[tuple[str, str], str] = {}
    prove_exit: int | None = None
    box: str | None = None
    driver: str | None = None
    first_group_ts: datetime | None = None
    last_boundary_ts: datetime | None = None
    stray_groups: list[tuple[datetime, str]] = []

    for ts, msg in events:
        gm = _GROUP_RE.search(msg)
        if gm:
            title = gm.group("title").strip()
            name = _canon_group_name(title, legacy)
            if name is not None:
                open_groups[name] = ts
                current_open_name = name
                if first_group_ts is None:
                    first_group_ts = ts
            else:
                stray_groups.append((ts, title))
            continue
        if _ENDGROUP_RE.search(msg):
            if current_open_name and current_open_name in open_groups:
                wall = (ts - open_groups[current_open_name]).total_seconds()
                closed.append(
                    {
                        "name": current_open_name,
                        "wall_s": round(wall, 3),
                        "rc": rc_by_name.get(current_open_name),
                    }
                )
                del open_groups[current_open_name]
                current_open_name = None
                # Only a CANONICAL group's own close can move the window's
                # end boundary -- a runner's own post-job cleanup group
                # (e.g. artifact upload) closing AFTER the driver's last
                # canonical group has no `current_open_name` to match here,
                # so it no longer silently drags `last_boundary_ts` (and
                # therefore `wall_s`) out to the whole job's duration.
                last_boundary_ts = ts
            continue
        rc_m = _GROUP_RC_RE.search(msg)
        if rc_m:
            rc_by_name[rc_m.group("name")] = int(rc_m.group("rc"))
            continue
        sha_m = _PROVE_SHA_RE.search(msg)
        if sha_m:
            git_sha = sha_m.group("sha")
            continue
        tup_m = _PROVE_TUPLE_RE.search(msg)
        if tup_m:
            tuples[(tup_m.group("crate"), tup_m.group("kind"))] = tup_m.group("features")
            continue
        exit_m = _PROVE_EXIT_RE.search(msg)
        if exit_m:
            prove_exit = int(exit_m.group("rc"))
            last_boundary_ts = ts
            continue
        if _DRIVER_EXIT_LINE_RE.search(msg):
            last_boundary_ts = ts
            continue
        dev_m = _DEVICE_CSV_RE.match(msg.strip())
        if dev_m and box is None and dev_m.group("name") != "name":
            box = dev_m.group("name").strip()
            driver = dev_m.group("driver").strip()

    if first_group_ts is None:
        raise ValueError("no `::group::` marker found in this log — not a gpu-prove job log")
    if last_boundary_ts is None or last_boundary_ts < first_group_ts:
        last_boundary_ts = events[-1][0]

    in_window_strays = [(t, title) for t, title in stray_groups if first_group_ts <= t <= last_boundary_ts]
    if in_window_strays:
        bad_ts, bad_title = in_window_strays[0]
        raise ValueError(
            f"non-canonical `##[group]{bad_title}` opened at {bad_ts.isoformat()}, INSIDE the "
            "measured window (first canonical group open through the last marker) -- the "
            "driver holds exactly one canonical group open at a time in this range, so an "
            "interleaved runner/other group here is a parse error, not a silent group"
        )

    max_gap = 0.0
    max_gap_after = None
    prev_ts = None
    prev_name = None
    for ts, msg in events:
        if ts < first_group_ts or ts > last_boundary_ts:
            continue
        if prev_ts is not None:
            gap = (ts - prev_ts).total_seconds()
            if gap > max_gap:
                max_gap = gap
                max_gap_after = prev_name
        prev_ts = ts
        gm = _GROUP_RE.search(msg)
        if gm:
            title = gm.group("title").strip()
            name = _canon_group_name(title, legacy)
            if name is not None:
                prev_name = name

    return {
        "groups": closed,
        "max_silent_gap_s": round(max_gap, 3),
        "max_silent_gap_after": max_gap_after,
        "git_sha": git_sha,
        "tuples": tuples,
        "prove_exit": prove_exit,
        "box": box,
        "driver": driver,
        "wall_s": round((last_boundary_ts - first_group_ts).total_seconds(), 3),
    }


def build_artifact(
    *,
    arch: str,
    run_id: str,
    job_id: str,
    log_text: str,
    legacy: bool,
    git_sha: str | None = None,
    outcome: str | None = None,
    box: str | None = None,
    driver: str | None = None,
    allow_incomplete: bool = False,
) -> dict:
    parsed = parse_log(log_text, legacy)
    # Computed unconditionally (legacy or current, explicit or auto-derived
    # --outcome): the WRONG TREE diagnostic can land in either log shape,
    # and an explicit `--outcome wrong-tree` still needs it below to
    # recover `expected`/`got` for the artifact's own `git_sha`/`proved_sha`.
    # A wrong-tree leg that never observed ANY `PROVE_SHA=` line (`got=none`)
    # has no `parsed["git_sha"]` at all -- its own `expected=` is the ONLY
    # honest source of `git_sha` short of an explicit `--git-sha`, so it is
    # tried BEFORE the "no git_sha derivable" refusal below, not after.
    wrong_tree_match = _WRONG_TREE_RE.search(log_text)
    git_sha = git_sha or parsed["git_sha"] or (wrong_tree_match.group("expected") if wrong_tree_match else None)
    box = box or parsed["box"]
    driver = driver or parsed["driver"]
    if not git_sha:
        raise ValueError("no git_sha derivable from the log (PROVE_SHA absent) — pass --git-sha")

    if legacy:
        if not outcome:
            raise ValueError("--legacy requires --outcome (not derivable from a legacy log)")
        surface = {"kind": "legacy-pre-d1", "expected_id": None}
        source = "run-metadata"
        groups = [{"name": g["name"], "wall_s": g["wall_s"], "rc": None} for g in parsed["groups"]]
    else:
        surface_pairs: dict[str, dict[str, list[str]]] = {}
        for (crate, kind), feat_text in parsed["tuples"].items():
            feats = [f for f in feat_text.split(",") if f]
            surface_pairs.setdefault(crate, {})[kind] = feats
        expected_id = prove_surface.expected_id(surface_pairs) if surface_pairs else None
        surface = {"kind": "prove-lane", "expected_id": expected_id}
        source = "job-log"
        groups = parsed["groups"]
        if outcome is None:
            # BLOCK B audit fix: the original `else` arm labelled EVERY
            # "PROVE_EXIT absent" case `budget-cut`, even a genuinely
            # healthy log whose gating rc's just happened to be 0 with no
            # PROVE_EXIT line for some other reason, and a log whose
            # gating groups were simply never populated (a truncated log,
            # not a cut) -- neither is actually a budget-cut. The real
            # discriminators are: did the remote reach its own final
            # `PROVE_EXIT=` line at all, do EVERY gating group's markers
            # exist, do they all read rc=0, and does the log carry the
            # driver's own BUDGET diagnostic (`rp_run_remote_watched`'s
            # own 124-cut evidence, printed only on a genuine ssh-status-124
            # cut with no PROVE_EXIT reached).
            gating_names = CURRENT_GROUP_NAMES - {"device", "bench"}
            group_names = {g["name"] for g in groups}
            all_gating_present = gating_names <= group_names
            all_gating_pass = all_gating_present and all(
                g["rc"] == 0 for g in groups if g["name"] in gating_names
            )
            has_prove_exit = parsed["prove_exit"] is not None
            has_budget_evidence = "BUDGET" in log_text
            has_no_progress_evidence = "NO PROGRESS" in log_text

            # BLOCK B9 audit fix: the driver's own WRONG TREE diagnostic is
            # tested FIRST, before `has_prove_exit` -- the bash-side final-
            # flush and absence arms (runpod_lib.sh's `rp_run_remote_watched`)
            # both leave a `PROVE_EXIT=0` line (and every gating marker
            # green) on a leg the driver itself refused as rc 77: the remote
            # script ran to completion believing it succeeded, but the
            # driver detected the wrong tree only in its OWN final flush,
            # AFTER the remote had already printed everything. Testing
            # `has_prove_exit` first would commit that leg as `healthy` at
            # the WRONG sha. The driver's 77 wins regardless of which
            # markers landed, exactly like `rp_run_remote_watched` itself:
            # the wrong-tree check runs unconditionally at exit and wins
            # over the remote's own reported success.
            if wrong_tree_match is not None:
                outcome = "wrong-tree"
            elif has_prove_exit:
                if parsed["prove_exit"] == 0 and all_gating_present and all_gating_pass:
                    outcome = "healthy"
                elif not all_gating_present:
                    # PROVE_EXIT was reached, but at least one gating group
                    # never got its own marker -- the log is TRUNCATED
                    # (missing content), not a real suite outcome. Never a
                    # silent guess: refuse without an explicit opt-in.
                    if not allow_incomplete:
                        raise ValueError(
                            "PROVE_EXIT is present but at least one gating group's marker is "
                            "missing -- the log looks TRUNCATED/incomplete, not a real outcome; "
                            "pass --allow-incomplete to write it as `log-incomplete` anyway, or "
                            "pass a real --outcome"
                        )
                    outcome = "log-incomplete"
                else:
                    # PROVE_EXIT present, every gating group has a marker,
                    # but at least one is non-zero (or PROVE_EXIT itself is
                    # non-zero) -- a genuine in-suite failure.
                    outcome = "suite-fail"
            elif has_budget_evidence:
                outcome = "budget-cut"
            elif has_no_progress_evidence:
                outcome = "watchdog-kill"
            else:
                # No WRONG TREE line, no PROVE_EXIT, no BUDGET line, no
                # NO-PROGRESS line -- the log gives no honest basis to pick
                # an outcome. Refuse rather than default to a specific
                # guess.
                raise ValueError(
                    "cannot auto-derive an outcome: no PROVE_EXIT= line, and neither a BUDGET, "
                    "NO PROGRESS, nor WRONG TREE driver diagnostic was found in the log -- pass "
                    "--outcome explicitly"
                )

    # esc-084/#454: a wrong-tree leg proved nothing -- record
    # the sha it EXPECTED as `git_sha` (the identity check happens outside
    # any proof group, so `parsed["git_sha"]` -- the observed `PROVE_SHA=`
    # line, if the mismatch itself was echoed -- is a `proved_sha`, never
    # this artifact's own `git_sha`). No SCHEMA_VERSION bump: `proved_sha`
    # is an additive optional key and `check_gpu_prove_timings.py` never
    # rejects an unrecognized key, only a recognized one in the wrong shape.
    proved_sha: str | None = None
    if outcome == "wrong-tree":
        if wrong_tree_match is not None:
            git_sha = wrong_tree_match.group("expected")
            got = wrong_tree_match.group("got")
            proved_sha = None if got == "none" else got
        elif not git_sha:
            raise ValueError(
                "outcome=wrong-tree but no WRONG TREE diagnostic was found in the log and no "
                "--git-sha was given to supply the expected sha"
            )

    artifact: dict = {
        "schema_version": SCHEMA_VERSION,
        "_doc": [
            "wall_s (healthy) / wall_lower_bound_s (cut/kill) is the WINDOW",
            "from the first `::group::` open to the last of {an",
            "`::endgroup::`, a `PROVE_EXIT=` line, a \"GPU prove suites",
            "exit=\" driver line} -- it deliberately EXCLUDES the GitHub",
            "runner's own provisioning/checkout overhead before the first",
            "group and the short post-job cleanup tail after the driver's",
            "own exit echo, neither of which belongs to the prove lane's",
            "own wall or silence accounting. `max_silent_gap_s` is computed",
            "over the SAME window -- EXCEPT on a `watchdog-kill` outcome,",
            "where the observed silence is, BY DEFINITION, right-censored",
            "at RP_INACTIVITY (the kill fires the moment that threshold is",
            "crossed, so the TRUE gap the process would have gone on to",
            "produce is unknown and >= this value): recorded as",
            "`silent_gap_lower_bound_s` instead, never as `max_silent_gap_s`",
            "-- R2 (check_gpu_prove_timings.py) consumes HEALTHY artifacts",
            "ONLY for exactly this reason, never a kill's own censored value.",
        ],
        "arch": arch,
        "box": box,
        "driver": driver,
        "run_id": run_id,
        "job_id": job_id,
        "git_sha": git_sha,
        "source": source,
        "surface": surface,
        "outcome": outcome,
        "max_silent_gap_after": parsed["max_silent_gap_after"],
        "groups": groups,
        "disposition": None,
    }
    if outcome == "wrong-tree":
        artifact["proved_sha"] = proved_sha
    if outcome == "watchdog-kill":
        artifact["silent_gap_lower_bound_s"] = parsed["max_silent_gap_s"]
    else:
        artifact["max_silent_gap_s"] = parsed["max_silent_gap_s"]
    if outcome == "healthy":
        artifact["wall_s"] = parsed["wall_s"]
    else:
        artifact["wall_lower_bound_s"] = parsed["wall_s"]
    return artifact


# --------------------------------------------------------------------------- #
# Self-test (BLOCK 1 audit fix): pins the CURRENT-mode marker-attribution
# path (a marker precedes its own `::endgroup::` by design) against both a
# fully healthy synthetic log and a genuine budget-cut one -- the exact
# shape whose bug (matching `closed` instead of the marker's own `name=`
# field) silently mislabeled every healthy leg as `budget-cut`.
# --------------------------------------------------------------------------- #
_JOB = "GPU prove on RunPod (sm_80)"
_STEP = "UNKNOWN STEP"


def _synth_log(lines: list[str]) -> str:
    """`lines`, one GH-raw-log line per entry, timestamped one second apart
    starting at a fixed epoch -- the exact `<job>\\t<step>\\t<ts> <msg>` shape
    `_LOG_LINE_RE` parses."""
    out = []
    base = datetime(2026, 1, 1, 0, 0, 0)
    for i, msg in enumerate(lines):
        stamp = (base + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
        out.append(f"{_JOB}\t{_STEP}\t{stamp} {msg}")
    return "\n".join(out) + "\n"


# The manifest's own `prove_lane.crates` declared (crate, kind) -> literal
# feature-text pairs, used to emit the SEVEN real PROVE_TUPLE echoes a
# healthy leg actually carries (round-2 audit advisory #3: the self-test's
# healthy log must carry them and assert `expected_id ==
# prove_surface.current_expected_id()`, never a bare synthetic sha).
_SELF_TEST_TUPLES = [
    ("jammi-server", "release"),
    ("jammi-ai", "test"),
    ("jammi-server", "test"),
    ("jammi-ai", "test"),  # engine-core-sweep -- same (crate,kind), same literal
    ("jammi-kernels", "default"),
    ("jammi-kernels", "test"),
    ("jammi-bench", "release"),
]
_SELF_TEST_GROUP_FOR_TUPLE = [
    "capability-surface-build",
    "capability-surface-build",
    "served-client-server-proof",
    "engine-core-sweep",
    "kernels-default",
    "kernels-cuda",
    "bench",
]


def _healthy_lines() -> list[str]:
    manifest = prove_surface.load_manifest()
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "a" * 40]
    tuple_idx = 0
    for name in (
        "capability-surface-build",
        "capability-surface-proof",
        "served-client-server-proof",
        "engine-core-sweep",
        "kernels-default",
        "kernels-cuda",
    ):
        lines.append(f"##[group]{name}")
        while tuple_idx < len(_SELF_TEST_GROUP_FOR_TUPLE) and _SELF_TEST_GROUP_FOR_TUPLE[tuple_idx] == name:
            crate, kind = _SELF_TEST_TUPLES[tuple_idx]
            feats = prove_surface.feature_text(prove_surface.expected(crate, kind, manifest))
            lines.append(f"PROVE_TUPLE crate={crate} kind={kind} features={feats}")
            tuple_idx += 1
        lines.append(f"PROVE_GROUP_RC name={name} rc=0")
        lines.append("##[endgroup]")
    lines.append("##[group]bench")
    crate, kind = _SELF_TEST_TUPLES[tuple_idx]
    feats = prove_surface.feature_text(prove_surface.expected(crate, kind, manifest))
    lines.append(f"PROVE_TUPLE crate={crate} kind={kind} features={feats}")
    lines.append("BENCH_EXIT=1")
    lines.append("PROVE_GROUP_RC name=bench rc=1")
    lines.append("##[endgroup]")
    lines.append("PROVE_EXIT=0")
    lines.append("=== GPU prove suites exit=0 (raw=0) ===")
    return lines


def _healthy_synth_log() -> str:
    return _synth_log(_healthy_lines())


def _synth_log_from_offsets(entries: list[tuple[float, str]]) -> str:
    """Same `<job>\\t<step>\\t<ts> <msg>` shape as `_synth_log`, but each
    line carries its OWN explicit second-offset from the fixed epoch
    instead of one-second-per-line -- needed to place a genuine 500+s gap
    (a real pod-provisioning wait) inside a fixture without inflating every
    OTHER line's spacing too."""
    out = []
    base = datetime(2026, 1, 1, 0, 0, 0)
    for offset, msg in entries:
        stamp = (base + timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%S.0000000Z")
        out.append(f"{_JOB}\t{_STEP}\t{stamp} {msg}")
    return "\n".join(out) + "\n"


def _healthy_synth_log_with_runner_preamble() -> str:
    """D5 measurement-scope regression fixture (esc-080..083 followup): a
    REALISTIC GitHub-runner preamble -- provisioning/checkout step groups,
    then the driver's own `waiting for SSH`/`SSH up` pair separated by a
    500+s pod-provisioning wait -- BEFORE this script's own first
    `::group::device`, exactly the shape a real D5 job log carries. The
    body after the preamble is byte-identical to `_healthy_lines()`'s own
    output, so `wall_s`/`max_silent_gap_s`/`groups[]` computed from this
    log must come out IDENTICAL to `_healthy_synth_log()`'s -- any
    difference means the preamble leaked into the measured window."""
    preamble: list[tuple[float, str]] = [
        (0.0, "##[group]Runner Image Provisioner"),
        (1.0, "Info: Provisioning is complete."),
        (2.0, "##[endgroup]"),
        (3.0, "##[group]Run actions/checkout@v4"),
        (4.0, "Cloning the repository"),
        (5.0, "##[endgroup]"),
        (6.0, "##[group]Run set +e"),
        (7.0, "deployed uuva3xc7ex3gmr on SECURE / NVIDIA H100 80GB HBM3; waiting for SSH (≤600s)..."),
        (520.0, "SSH up on uuva3xc7ex3gmr"),
    ]
    preamble_end = preamble[-1][0]
    entries = list(preamble)
    for i, msg in enumerate(_healthy_lines()):
        entries.append((preamble_end + 1 + i, msg))
    return _synth_log_from_offsets(entries)


def _cut_synth_log() -> str:
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "b" * 40]
    for name in ("capability-surface-build", "capability-surface-proof", "served-client-server-proof"):
        lines.append(f"##[group]{name}")
        lines.append(f"PROVE_GROUP_RC name={name} rc=0")
        lines.append("##[endgroup]")
    # engine-core-sweep is cut mid-flight -- opened, never closed, no marker.
    lines.append("##[group]engine-core-sweep")
    lines.append("Compiling jammi-kernels v0.48.0")
    # The driver's own BUDGET diagnostic (rp_run_remote_watched's real
    # output on a genuine ssh-status-124 cut with no PROVE_EXIT reached) --
    # BLOCK B audit fix: without this evidence in the log, `budget-cut` can
    # no longer be auto-derived at all (the producer now refuses to guess).
    lines.append('=== GPU prove: BUDGET (RP_TIMEOUT=6000s) cut group "engine-core-sweep"; groups: [capability-surface-build:0,capability-surface-proof:0,served-client-server-proof:0] ===')
    return _synth_log(lines)


def _suite_fail_synth_log() -> str:
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "c" * 40]
    for name, rc in (
        ("capability-surface-build", 0),
        ("capability-surface-proof", 0),
        ("served-client-server-proof", 1),
        ("engine-core-sweep", 0),
        ("kernels-default", 0),
        ("kernels-cuda", 0),
    ):
        lines.append(f"##[group]{name}")
        lines.append(f"PROVE_GROUP_RC name={name} rc={rc}")
        lines.append("##[endgroup]")
    lines.append("PROVE_EXIT=1")
    lines.append("=== GPU prove suites exit=1 (raw=1) ===")
    return _synth_log(lines)


def _log_incomplete_synth_log() -> str:
    # PROVE_EXIT was reached, but kernels-cuda's own marker never landed --
    # a TRUNCATED log (e.g. a producer bug, a partial download), never a
    # genuine outcome to guess at.
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "d" * 40]
    for name in ("capability-surface-build", "capability-surface-proof", "served-client-server-proof", "engine-core-sweep", "kernels-default"):
        lines.append(f"##[group]{name}")
        lines.append(f"PROVE_GROUP_RC name={name} rc=0")
        lines.append("##[endgroup]")
    lines.append("PROVE_EXIT=0")
    return _synth_log(lines)


def _watchdog_kill_synth_log() -> str:
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "e" * 40]
    lines.append("##[group]kernels-cuda")
    lines.append("Compiling jammi-kernels v0.48.0")
    lines.append('=== GPU prove: NO PROGRESS for 600s in group "kernels-cuda"; groups: [] ===')
    return _synth_log(lines)


def _wrong_tree_synth_log(observed_prove_sha: str | None) -> str:
    """`device` opens/closes first (the real driver's own group order --
    it runs before the clone), THEN the identity check fires -- outside any
    proof group, matching `runpod_gpu_prove.sh`'s own heredoc order.
    `observed_prove_sha=None` reproduces the "absence" case (session ended
    with no `PROVE_SHA=` line at all)."""
    lines = [
        "##[group]device",
        "name, compute_cap, driver_version",
        "NVIDIA A100 80GB PCIe, 8.0, 570.195.03",
        "CUDA_COMPUTE_CAP=80",
        "##[endgroup]",
    ]
    if observed_prove_sha is not None:
        lines.append(f"PROVE_SHA={observed_prove_sha}")
    got = observed_prove_sha or "none"
    lines.append(f'=== GPU prove: WRONG TREE expected={"f" * 40} got={got} ===')
    return _synth_log(lines)


def _wrong_tree_healthy_synth_log(observed_prove_sha: str | None) -> str:
    """BLOCK B9 audit fix regression fixture: the WRONG-TREE-in-the-final-
    flush shape -- the remote script ran ALL THE WAY to a self-reported
    healthy finish (every gating group green, `PROVE_EXIT=0`) and the
    driver's own identity check only caught the mismatch (or the absence)
    in its final flush, AFTER the remote had already printed its own
    success. `observed_prove_sha=None` reproduces the absence sub-case: no
    `PROVE_SHA=` line was ever echoed, yet the rest of the leg still looks
    entirely healthy on its own terms. Either way `build_artifact` must
    still resolve `wrong-tree`, never `healthy` -- the driver's 77 wins
    regardless of which markers landed."""
    lines = list(_healthy_lines())
    if observed_prove_sha is None:
        # Drop the PROVE_SHA= echo the base healthy log carries -- the
        # absence sub-case never observed one at all.
        lines = [ln for ln in lines if not ln.startswith("PROVE_SHA=")]
        got = "none"
    else:
        got = observed_prove_sha
    lines.append(f'=== GPU prove: WRONG TREE expected={"f" * 40} got={got} ===')
    return _synth_log(lines)


def _self_test() -> int:
    failures: list[str] = []
    total = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal total
        total += 1
        print(f"self-test[{name}]: " + ("ok" if cond else f"FAIL -- {detail}"))
        if not cond:
            failures.append(name)

    healthy_artifact = build_artifact(
        arch="sm_80", run_id="1", job_id="1", log_text=_healthy_synth_log(), legacy=False
    )
    check("healthy-outcome", healthy_artifact["outcome"] == "healthy", f"{healthy_artifact['outcome']}")
    check("healthy-has-wall-s", "wall_s" in healthy_artifact and isinstance(healthy_artifact["wall_s"], float), "")
    check("healthy-no-wall-lower-bound", "wall_lower_bound_s" not in healthy_artifact, "")
    gating = {g["name"]: g["rc"] for g in healthy_artifact["groups"] if g["name"] in CURRENT_GROUP_NAMES - {"device", "bench"}}
    check(
        "healthy-all-six-gating-rcs-zero",
        len(gating) == 6 and all(rc == 0 for rc in gating.values()),
        f"{gating}",
    )
    bench_group = next(g for g in healthy_artifact["groups"] if g["name"] == "bench")
    check("healthy-bench-rc-recorded-non-gating", bench_group["rc"] == 1, f"{bench_group}")
    # Round-2 audit advisory #3: the healthy self-test log carries the real
    # PROVE_TUPLE echoes, and the resulting expected_id must equal the
    # CURRENT manifest's own canonicalization -- never a bare synthetic sha
    # standing in for a real surface fingerprint.
    check(
        "healthy-expected-id-matches-current-manifest",
        healthy_artifact["surface"]["expected_id"] == prove_surface.current_expected_id(),
        f"{healthy_artifact['surface']['expected_id']} vs {prove_surface.current_expected_id()}",
    )

    cut_artifact = build_artifact(arch="sm_80", run_id="2", job_id="2", log_text=_cut_synth_log(), legacy=False)
    check("cut-outcome", cut_artifact["outcome"] == "budget-cut", f"{cut_artifact['outcome']}")
    check("cut-has-wall-lower-bound", "wall_lower_bound_s" in cut_artifact, "")
    check("cut-no-wall-s", "wall_s" not in cut_artifact, "")
    cut_names = {g["name"] for g in cut_artifact["groups"]}
    check("cut-engine-core-sweep-not-closed", "engine-core-sweep" not in cut_names, f"{cut_names}")

    # BLOCK B audit fix -- the three-way `else`-arm split.
    suite_fail_artifact = build_artifact(arch="sm_80", run_id="3", job_id="3", log_text=_suite_fail_synth_log(), legacy=False)
    check("suite-fail-outcome", suite_fail_artifact["outcome"] == "suite-fail", f"{suite_fail_artifact['outcome']}")

    try:
        build_artifact(arch="sm_80", run_id="4", job_id="4", log_text=_log_incomplete_synth_log(), legacy=False)
        check("log-incomplete-refused-without-flag", False, "expected a ValueError")
    except ValueError:
        check("log-incomplete-refused-without-flag", True)
    incomplete_artifact = build_artifact(
        arch="sm_80", run_id="4", job_id="4", log_text=_log_incomplete_synth_log(), legacy=False, allow_incomplete=True
    )
    check("log-incomplete-outcome-with-flag", incomplete_artifact["outcome"] == "log-incomplete", f"{incomplete_artifact['outcome']}")

    watchdog_artifact = build_artifact(arch="sm_80", run_id="5", job_id="5", log_text=_watchdog_kill_synth_log(), legacy=False)
    check("watchdog-kill-outcome", watchdog_artifact["outcome"] == "watchdog-kill", f"{watchdog_artifact['outcome']}")

    # esc-084/#454: wrong-tree, both shapes -- a real (wrong)
    # PROVE_SHA was observed, and the absence case (no PROVE_SHA at all).
    wrong_tree_artifact = build_artifact(
        arch="sm_80", run_id="6", job_id="6", log_text=_wrong_tree_synth_log("a" * 40), legacy=False
    )
    check("wrong-tree-outcome", wrong_tree_artifact["outcome"] == "wrong-tree", f"{wrong_tree_artifact['outcome']}")
    check("wrong-tree-git-sha-is-expected", wrong_tree_artifact["git_sha"] == "f" * 40, wrong_tree_artifact["git_sha"])
    check("wrong-tree-proved-sha-is-got", wrong_tree_artifact["proved_sha"] == "a" * 40, wrong_tree_artifact.get("proved_sha"))
    check("wrong-tree-has-wall-lower-bound", "wall_lower_bound_s" in wrong_tree_artifact, "")

    wrong_tree_absent_artifact = build_artifact(
        arch="sm_86", run_id="7", job_id="7", log_text=_wrong_tree_synth_log(None), legacy=False
    )
    check("wrong-tree-absent-outcome", wrong_tree_absent_artifact["outcome"] == "wrong-tree", f"{wrong_tree_absent_artifact['outcome']}")
    check("wrong-tree-absent-git-sha-is-expected", wrong_tree_absent_artifact["git_sha"] == "f" * 40, wrong_tree_absent_artifact["git_sha"])
    check("wrong-tree-absent-proved-sha-is-none", wrong_tree_absent_artifact["proved_sha"] is None, wrong_tree_absent_artifact.get("proved_sha"))

    # BLOCK B9 audit fix regression: a log that ALSO carries PROVE_EXIT=0
    # and every gating group green (the final-flush shape) must still
    # resolve `wrong-tree`, never `healthy` -- the driver's 77 wins
    # regardless of which other markers landed in the same log.
    wrong_tree_healthy_artifact = build_artifact(
        arch="sm_80", run_id="8", job_id="8", log_text=_wrong_tree_healthy_synth_log("a" * 40), legacy=False
    )
    check(
        "wrong-tree-wins-over-healthy-markers-outcome",
        wrong_tree_healthy_artifact["outcome"] == "wrong-tree",
        f"{wrong_tree_healthy_artifact['outcome']}",
    )
    check(
        "wrong-tree-wins-over-healthy-markers-git-sha-is-expected",
        wrong_tree_healthy_artifact["git_sha"] == "f" * 40,
        wrong_tree_healthy_artifact["git_sha"],
    )
    check(
        "wrong-tree-wins-over-healthy-markers-proved-sha-is-got",
        wrong_tree_healthy_artifact["proved_sha"] == "a" * 40,
        wrong_tree_healthy_artifact.get("proved_sha"),
    )

    # Same shape, absence sub-case: PROVE_EXIT=0 and every gating group
    # green, but NO PROVE_SHA= line was ever echoed -- still `wrong-tree`,
    # `proved_sha` null.
    wrong_tree_healthy_absent_artifact = build_artifact(
        arch="sm_86", run_id="9", job_id="9", log_text=_wrong_tree_healthy_synth_log(None), legacy=False
    )
    check(
        "wrong-tree-healthy-absent-outcome",
        wrong_tree_healthy_absent_artifact["outcome"] == "wrong-tree",
        f"{wrong_tree_healthy_absent_artifact['outcome']}",
    )
    check(
        "wrong-tree-healthy-absent-git-sha-is-expected",
        wrong_tree_healthy_absent_artifact["git_sha"] == "f" * 40,
        wrong_tree_healthy_absent_artifact["git_sha"],
    )
    check(
        "wrong-tree-healthy-absent-proved-sha-is-none",
        wrong_tree_healthy_absent_artifact["proved_sha"] is None,
        wrong_tree_healthy_absent_artifact.get("proved_sha"),
    )

    # D5 measurement-scope fix: a runner preamble (provisioning/checkout
    # groups + a 500+s SSH-wait gap) BEFORE `::group::device` must not move
    # `wall_s`/`max_silent_gap_s`, and `groups[]` must carry ONLY canonical
    # names -- never the runner's own step groups.
    preamble_artifact = build_artifact(
        arch="sm_80", run_id="6", job_id="6", log_text=_healthy_synth_log_with_runner_preamble(), legacy=False
    )
    check(
        "runner-preamble-wall-s-unchanged",
        preamble_artifact["wall_s"] == healthy_artifact["wall_s"],
        f"{preamble_artifact['wall_s']} vs {healthy_artifact['wall_s']}",
    )
    check(
        "runner-preamble-max-silent-gap-unchanged",
        preamble_artifact["max_silent_gap_s"] == healthy_artifact["max_silent_gap_s"],
        f"{preamble_artifact['max_silent_gap_s']} vs {healthy_artifact['max_silent_gap_s']}",
    )
    preamble_group_names = {g["name"] for g in preamble_artifact["groups"]}
    check(
        "runner-preamble-groups-canonical-only",
        preamble_group_names <= CURRENT_GROUP_NAMES,
        f"{preamble_group_names}",
    )
    check(
        "runner-preamble-groups-same-as-no-preamble",
        preamble_group_names == {g["name"] for g in healthy_artifact["groups"]},
        f"{preamble_group_names} vs {[g['name'] for g in healthy_artifact['groups']]}",
    )

    # A non-canonical `##[group]` INSIDE the measured window is a parse
    # error, not a silently-dropped group (BLOCK C follow-up: the driver's
    # own convention never interleaves a foreign group between canonical
    # ones, so seeing one there means the log itself is suspect).
    stray_inside_lines = _healthy_lines()
    device_close_idx = stray_inside_lines.index("##[endgroup]")
    stray_inside_lines = (
        stray_inside_lines[: device_close_idx + 1]
        + ["##[group]Post job cleanup", "##[endgroup]"]
        + stray_inside_lines[device_close_idx + 1 :]
    )
    try:
        parse_log(_synth_log(stray_inside_lines), legacy=False)
        check("stray-group-inside-window-rejected", False, "expected a ValueError")
    except ValueError as e:
        check("stray-group-inside-window-rejected", "INSIDE the" in str(e), str(e))

    if failures:
        print(f"self-test: FAIL ({len(failures)}/{total} failing): {failures}", file=sys.stderr)
        return 1
    print(f"self-test: all {total} checks passed")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return _self_test()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_log", type=Path)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--git-sha")
    ap.add_argument(
        "--outcome",
        choices=["healthy", "budget-cut", "watchdog-kill", "suite-fail", "capacity", "log-incomplete", "wrong-tree"],
    )
    ap.add_argument("--box")
    ap.add_argument("--driver")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit writing a `log-incomplete` artifact (PROVE_EXIT reached but a gating "
        "group's marker is missing) -- refused by default so a truncated log is never "
        "silently written as if it were a real outcome.",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    text = args.job_log.read_text(encoding="utf-8", errors="replace")
    try:
        artifact = build_artifact(
            arch=args.arch,
            run_id=args.run_id,
            job_id=args.job_id,
            log_text=text,
            legacy=args.legacy,
            git_sha=args.git_sha,
            outcome=args.outcome,
            box=args.box,
            driver=args.driver,
            allow_incomplete=args.allow_incomplete,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out = args.out or (ARTIFACT_DIR / f"{args.run_id}-{args.arch}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
