#!/usr/bin/env python3
"""The ladder's budgets, read off its committed verdicts.

`crates/jammi-bench/budgets.json` holds every measured bound a ladder rule
judges against, one entry per `(workload, edge, rule)` naming the artifact
its bound was read from. This script is the one derivation of that table:
each committed `ladder_verdict.json` a GPU session produced supplies, per
edge, the bound its measurement establishes, and a later run of the same
edge is held to it —

  * `overhead_budget`        a layer's or kernel arm's cost: the upper end of the
                              measured `upper ÷ lower` interval of medians;
  * `speed_non_inferiority`  a framework edge's bar: the lower end of the measured
                              `lower ÷ upper` interval (the reciprocal of the cost's
                              upper end);
  * `host_memory_ratio`, `device_memory_ratio`
                              the measured peak-memory ratios;
  * `fixed_cost`, `per_work_cost`
                              a size sweep's fitted fixed-cost work equivalent and
                              per-work ratio;
  * `time_to_quality`        the measured `lower ÷ upper` time-to-quality ratio.

Every bound is rounded outward to three decimals, so a re-run that lands on the
measurement itself passes. A rule a verdict did not measure gets no entry, and
stays `UNBUDGETED`.

Usage:
    budgets_from_verdicts.py --out crates/jammi-bench/budgets.json ARTIFACT.json...
    budgets_from_verdicts.py --check crates/jammi-bench/budgets.json [ARTIFACT.json...]
        (with no artifacts named, the table's own `measured_from` entries are the sources)
    budgets_from_verdicts.py --self-test

Artifact paths are given relative to the repository root, and are recorded that
way in `measured_from`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

DECIMALS = 3
# The verdict's kind of difference that names a layer's cost, and the one that
# names a framework's bar.
LAYER_KINDS = ("layer",)
FRAMEWORK_KIND = "framework"


def ceil_out(x: float) -> float:
    return math.ceil(x * 10**DECIMALS) / 10**DECIMALS


def floor_out(x: float) -> float:
    return math.floor(x * 10**DECIMALS) / 10**DECIMALS


def interval_bounds(interval: dict) -> tuple[float, float]:
    for lo, hi in (("lo", "hi"), ("low", "high"), ("lower", "upper"), ("min", "max")):
        if lo in interval and hi in interval:
            return float(interval[lo]), float(interval[hi])
    raise ValueError(f"interval without recognisable bounds: {sorted(interval)}")


def entries_of(verdict: dict, measured_from: str) -> list[dict]:
    out: list[dict] = []
    workload = verdict["workload"]

    def entry(edge: str, rule: str, bound: float) -> None:
        out.append(
            {
                "workload": workload,
                "edge": edge,
                "rule": rule,
                "bound": bound,
                "measured_from": measured_from,
            }
        )

    for edge in verdict.get("edges", []):
        name = edge["edge"]
        kind = (edge.get("difference") or {}).get("kind")
        speed = edge.get("speed")
        if speed and speed.get("cost"):
            _, hi = interval_bounds(speed["cost"]["interval"])
            if kind in LAYER_KINDS:
                entry(name, "overhead_budget", ceil_out(hi))
            elif kind == FRAMEWORK_KIND and hi > 0:
                entry(name, "speed_non_inferiority", floor_out(1.0 / hi))
            shape = speed.get("shape")
            if shape:
                entry(name, "fixed_cost", ceil_out(float(shape["fixed_work_equivalent"])))
                entry(name, "per_work_cost", ceil_out(float(shape["per_work_ratio"])))
            ttq = speed.get("time_to_quality")
            if ttq and ttq.get("ratio") is not None:
                entry(name, "time_to_quality", floor_out(float(ttq["ratio"])))
        space = edge.get("space")
        if space:
            if space.get("host_ratio") is not None:
                entry(name, "host_memory_ratio", ceil_out(float(space["host_ratio"])))
            if space.get("device_ratio") is not None:
                entry(name, "device_memory_ratio", ceil_out(float(space["device_ratio"])))
    return out


def derive(repo_root: Path, artifacts: list[str]) -> dict:
    entries: list[dict] = []
    for rel in artifacts:
        verdict = json.loads((repo_root / rel).read_text())
        entries.extend(entries_of(verdict, rel))
    entries.sort(key=lambda e: (e["workload"], e["edge"], e["rule"]))
    return {"budgets": entries}


def render(table: dict) -> str:
    return json.dumps(table, indent=2) + "\n"


def self_test() -> int:
    verdict = {
        "workload": "encode",
        "edges": [
            {
                "edge": "torch -> direct",
                "difference": {"kind": "framework", "reference": "PyTorch"},
                "speed": {"cost": {"of_minima": 0.5, "of_medians": 0.52, "interval": {"lo": 0.48, "hi": 0.55}}},
                "space": {"host_ratio": 1.1234, "device_ratio": None},
            },
            {
                "edge": "direct -> plan",
                "difference": {"kind": "layer", "name": "plan"},
                "speed": {
                    "cost": {"of_minima": 1.02, "of_medians": 1.03, "interval": {"lo": 1.011, "hi": 1.0421}},
                    "shape": {"fixed_work_equivalent": 12.34, "per_work_ratio": 1.0001},
                },
                "space": {"host_ratio": 1.0, "device_ratio": 0.99},
            },
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "v.json").write_text(json.dumps(verdict))
        got = derive(root, ["v.json"])["budgets"]
    want = [
        ("encode", "direct -> plan", "device_memory_ratio", 0.99),
        ("encode", "direct -> plan", "fixed_cost", 12.34),
        ("encode", "direct -> plan", "host_memory_ratio", 1.0),
        ("encode", "direct -> plan", "overhead_budget", 1.043),
        ("encode", "direct -> plan", "per_work_cost", 1.001),
        ("encode", "torch -> direct", "host_memory_ratio", 1.124),
        ("encode", "torch -> direct", "speed_non_inferiority", 1.818),
    ]
    have = [(e["workload"], e["edge"], e["rule"], e["bound"]) for e in got]
    if have != want:
        print("self-test FAILED\n want:", want, "\n have:", have, file=sys.stderr)
        return 1
    assert all(e["measured_from"] == "v.json" for e in got)
    print("budgets_from_verdicts self-test: OK — every rule kind derives its bound, rounded outward, sorted")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="write the table here")
    parser.add_argument("--check", help="assert this committed table equals the derivation")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("artifacts", nargs="*", help="verdict artifacts, repo-relative")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.artifacts and not args.check:
        parser.error("name at least one verdict artifact")
    table = derive(Path(args.repo_root), args.artifacts) if args.artifacts else {"budgets": []}
    if args.out:
        Path(args.out).write_text(render(table))
        print(f"wrote {len(table['budgets'])} budgets to {args.out}")
        return 0
    if args.check:
        committed = Path(args.check).read_text()
        if not args.artifacts:
            # The table names its own sources: re-derive from exactly those.
            artifacts = sorted({e["measured_from"] for e in json.loads(committed)["budgets"]})
            table = derive(Path(args.repo_root), artifacts)
            args.artifacts = artifacts
        if committed != render(table):
            print(f"::error::{args.check} does not equal its derivation from the named artifacts — regenerate with --out", file=sys.stderr)
            return 1
        print(f"budgets: {args.check} follows its {len(args.artifacts)} artifact(s)")
        return 0
    print(render(table), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
