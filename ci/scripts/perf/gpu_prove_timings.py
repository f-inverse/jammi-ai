#!/usr/bin/env python3
"""GPU-prove-lane timing/outcome artifact producer (esc-080/esc-083).

Parses a `gpu-prove.yml`/`_gpu-prove-gate.yml` job log (the raw, tab-
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
_PROVE_SHA_RE = re.compile(r"PROVE_SHA=(?P<sha>[0-9a-f]+)")
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


def parse_log(text: str, legacy: bool) -> dict:
    """Returns {groups: [{name, wall_s, rc}], max_silent_gap_s,
    max_silent_gap_after, git_sha (or None), tuples: {(crate,kind): feat},
    prove_exit (or None), box, driver, wall_s}.

    The WALL/silence window is bounded to [first `::group::` open, last of
    {an `::endgroup::`, a `PROVE_EXIT=` line, a "GPU prove suites exit="
    driver line}] -- NOT the whole job log, which also carries the GitHub
    runner's own provisioning/checkout overhead (several minutes) before
    this script's first `::group::device` line, and a short post-job
    cleanup tail after the driver's own exit echo. Neither belongs to the
    prove lane's own wall or silence accounting.
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

    for ts, msg in events:
        gm = _GROUP_RE.search(msg)
        if gm:
            title = gm.group("title").strip()
            name = _legacy_group_name(title) if legacy else title
            if name is not None:
                open_groups[name] = ts
                current_open_name = name
                if first_group_ts is None:
                    first_group_ts = ts
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
            name = _legacy_group_name(title) if legacy else title
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
) -> dict:
    parsed = parse_log(log_text, legacy)
    git_sha = git_sha or parsed["git_sha"]
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
            group_names = {g["name"] for g in groups}
            gating_ok = CURRENT_GROUP_NAMES - {"device", "bench"} <= group_names and all(
                g["rc"] == 0 for g in groups if g["name"] in CURRENT_GROUP_NAMES - {"device", "bench"}
            )
            if parsed["prove_exit"] == 0 and gating_ok:
                outcome = "healthy"
            elif parsed["prove_exit"] not in (None, 0):
                outcome = "suite-fail"
            else:
                outcome = "budget-cut"  # a cut/kill with no PROVE_EXIT observed

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
            "over the SAME window.",
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
        "max_silent_gap_s": parsed["max_silent_gap_s"],
        "max_silent_gap_after": parsed["max_silent_gap_after"],
        "groups": groups,
        "disposition": None,
    }
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


def _healthy_synth_log() -> str:
    lines = ["##[group]device", "name, compute_cap, driver_version", "NVIDIA A100 80GB PCIe, 8.0, 570.195.03", "CUDA_COMPUTE_CAP=80", "##[endgroup]"]
    lines += ["PROVE_SHA=" + "a" * 40]
    for name in (
        "capability-surface-build",
        "capability-surface-proof",
        "served-client-server-proof",
        "engine-core-sweep",
        "kernels-default",
        "kernels-cuda",
    ):
        lines.append(f"##[group]{name}")
        lines.append(f"PROVE_GROUP_RC name={name} rc=0")
        lines.append("##[endgroup]")
    lines.append("##[group]bench")
    lines.append("BENCH_EXIT=1")
    lines.append("PROVE_GROUP_RC name=bench rc=1")
    lines.append("##[endgroup]")
    lines.append("PROVE_EXIT=0")
    lines.append("=== GPU prove suites exit=0 (raw=0) ===")
    return _synth_log(lines)


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

    cut_artifact = build_artifact(arch="sm_80", run_id="2", job_id="2", log_text=_cut_synth_log(), legacy=False)
    check("cut-outcome", cut_artifact["outcome"] == "budget-cut", f"{cut_artifact['outcome']}")
    check("cut-has-wall-lower-bound", "wall_lower_bound_s" in cut_artifact, "")
    check("cut-no-wall-s", "wall_s" not in cut_artifact, "")
    cut_names = {g["name"] for g in cut_artifact["groups"]}
    check("cut-engine-core-sweep-not-closed", "engine-core-sweep" not in cut_names, f"{cut_names}")

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
    ap.add_argument("--outcome", choices=["healthy", "budget-cut", "watchdog-kill", "suite-fail", "capacity"])
    ap.add_argument("--box")
    ap.add_argument("--driver")
    ap.add_argument("--legacy", action="store_true")
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
