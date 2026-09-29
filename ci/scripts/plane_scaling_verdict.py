#!/usr/bin/env python3
"""The compute plane's scaling test: its legs laid out for the ladder, and
the pre-registered decision read off the ladder's verdicts.

The test and its pass bar are D15 of
`docs/plans/68-compute-tier-substrate/units/DIST-DATA-PLANE.md`. This program
decides nothing the ladder did not measure: `assemble` files the sessions'
legs where `jammi-bench ladder encode` reads them, and `decide` applies D15's
four conditions to the numbers each K's verdict carries.

    plane_scaling_verdict.py assemble DEST
        DEST/edge holds the `plan-partitioned` and `placed` legs of the edge
        session; DEST/direct-K<k> the `plan-partitioned` and `shape-d` legs of
        the direct session at a fleet of k. Writes DEST/ladder-K<k> (the edge
        legs and the fleet's `shape-d` legs) and DEST/ladder-K<k>/direct (the
        direct session's legs), refusing any `shape-d` leg whose recorded
        compute tier is not k live executors.
    plane_scaling_verdict.py decide DEST
        Read DEST/ladder-K<k>/verdict/ladder_verdict.json for every k and print
        D15's reading of each; exit 0 on PASS, 1 on FAIL, 2 when a verdict a
        judged k needs is missing.
    plane_scaling_verdict.py --self-test
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

FLEETS = (1, 2, 4)
# D15's bar, by fleet size: the composed per-row speedup and the direct
# pair's end-to-end speedup at the conservative end of its interval. K = 1 is
# reported, not judged.
BAR = {2: (1.7, 1.4), 4: (3.0, 2.2)}
EDGES = ("plan-partitioned -> placed", "placed -> shape-d")
TIER = re.compile(r"compute tier: (\d+) live executors")


def leg_rung(path: Path) -> str:
    return path.name.split("__", 1)[0]


def compute_tier(leg: Path) -> int | None:
    ran_on = json.loads(leg.read_text())["tiers"]["encode_step"].get("ran_on") or {}
    for line in ran_on.get("evidence", []):
        if match := TIER.search(line):
            return int(match.group(1))
    return None


def copy_session(src: Path, dst: Path, keep) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.iterdir()):
        if path.is_file() and keep(path):
            shutil.copy2(path, dst / path.name)


def assemble(dest: Path) -> list[str]:
    findings: list[str] = []
    edge = dest / "edge"
    for k in FLEETS:
        direct = dest / f"direct-K{k}"
        if not direct.is_dir():
            findings.append(f"no direct session at K = {k} under {direct}")
            continue
        placed = sorted(direct.glob("shape-d__*.json"))
        local = sorted(direct.glob("plan-partitioned__*.json"))
        if not placed or len(placed) != len(local):
            findings.append(
                f"{direct}: {len(placed)} shape-d legs beside {len(local)} plan-partitioned legs — "
                "a direct session files one of each per unit and take"
            )
        for leg in placed:
            tier = compute_tier(leg)
            if tier != k:
                findings.append(f"{leg}: served against a compute tier of {tier}, not {k}")
        ladder = dest / f"ladder-K{k}"
        copy_session(edge, ladder, lambda p: leg_rung(p) in ("plan-partitioned", "placed"))
        copy_session(direct, ladder, lambda p: leg_rung(p) == "shape-d")
        copy_session(direct, ladder / "direct", lambda p: leg_rung(p) in ("plan-partitioned", "shape-d"))
    return findings


def reading(verdict: dict) -> dict:
    edges = {e["edge"]: e for e in verdict["edges"]}
    missing = [name for name in EDGES if name not in edges]
    if missing:
        return {"error": f"the verdict has no edge {missing}"}
    outcomes_equal = all(
        (e.get("outcome") or {}).get("units_equal") == (e.get("outcome") or {}).get("units")
        and (e.get("outcome") or {}).get("units")
        for e in edges.values()
    )
    shapes = [(edges[name].get("speed") or {}).get("shape") for name in EDGES]
    per_work = None
    if all(shapes):
        product = 1.0
        for shape in shapes:
            product *= float(shape["per_work_ratio"])
        per_work = 1.0 / product
    telescoping = verdict.get("telescoping")
    direct = None
    if telescoping:
        direct = 1.0 / float(telescoping["direct"]["interval"]["upper"])
    return {
        "status": verdict["status"],
        "outcomes_equal": outcomes_equal,
        "per_work_speedup": per_work,
        "direct_speedup": direct,
    }


def judged(k: int, r: dict) -> list[str]:
    per_work_bar, direct_bar = BAR[k]
    failed = []
    if "error" in r:
        return [r["error"]]
    if not r["outcomes_equal"]:
        failed.append("an edge's outcome digests differ")
    if r["status"] == "INVALID":
        failed.append("the verdict is refused")
    if r["per_work_speedup"] is None or r["per_work_speedup"] < per_work_bar:
        failed.append(f"per-row speedup {r['per_work_speedup']} below {per_work_bar}")
    if r["direct_speedup"] is None or r["direct_speedup"] < direct_bar:
        failed.append(f"end-to-end speedup {r['direct_speedup']} below {direct_bar}")
    return failed


def decide(dest: Path) -> int:
    verdicts = {}
    for k in FLEETS:
        path = dest / f"ladder-K{k}" / "verdict" / "ladder_verdict.json"
        if not path.is_file():
            if k in BAR:
                print(f"::error::no verdict at K = {k}: {path}", file=sys.stderr)
                return 2
            continue
        verdicts[k] = reading(json.loads(path.read_text()))
    passed = True
    for k, r in sorted(verdicts.items()):
        line = f"K = {k}: {json.dumps(r)}"
        if k in BAR:
            failed = judged(k, r)
            passed = passed and not failed
            line += "  PASS" if not failed else f"  FAIL ({'; '.join(failed)})"
        else:
            line += "  (reported, not judged)"
        print(line)
    print("D15:", "PASS — the plane keeps its place" if passed else "FAIL — the plane is removed")
    return 0 if passed else 1


def self_test() -> int:
    def verdict(pw: tuple[float, float], direct_upper: float, status="GREEN", equal=True) -> dict:
        outcome = {"kind": "digest", "units_equal": 2 if equal else 1, "units": 2}
        return {
            "status": status,
            "edges": [
                {"edge": name, "outcome": outcome, "speed": {"shape": {"per_work_ratio": w}}}
                for name, w in zip(EDGES, pw)
            ],
            "telescoping": {"direct": {"interval": {"lower": direct_upper * 0.9, "upper": direct_upper}}},
        }

    cases = [
        ("a fleet that scales", 2, verdict((1.0, 0.55), 0.65), True),
        ("a fleet whose plane eats the gain", 2, verdict((1.1, 0.6), 0.8), False),
        ("digests that differ", 4, verdict((1.0, 0.3), 0.4, equal=False), False),
        ("a refused verdict", 4, verdict((1.0, 0.3), 0.4, status="INVALID"), False),
        ("a fleet of four that scales", 4, verdict((1.02, 0.31), 0.44), True),
    ]
    for label, k, v, want in cases:
        if (not judged(k, reading(v))) != want:
            print(f"self-test FAILED: {label}: {judged(k, reading(v))}", file=sys.stderr)
            return 1
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)

        def leg(dir_: Path, name: str, tier: int | None) -> None:
            dir_.mkdir(parents=True, exist_ok=True)
            evidence = [] if tier is None else [f"compute tier: {tier} live executors: a; b"]
            (dir_ / name).write_text(json.dumps({"tiers": {"encode_step": {"ran_on": {"evidence": evidence}}}}))

        leg(dest / "edge", "plan-partitioned__rows16384__r1.json", None)
        leg(dest / "edge", "placed__rows16384__r1.json", None)
        for k in FLEETS:
            leg(dest / f"direct-K{k}", "plan-partitioned__rows16384__r1.json", None)
            leg(dest / f"direct-K{k}", "shape-d__rows16384__r1.json", k if k != 4 else 3)
        findings = assemble(dest)
        if len(findings) != 1 or "not 4" not in findings[0]:
            print(f"self-test FAILED: a mis-sized fleet: {findings}", file=sys.stderr)
            return 1
        (dest / "direct-K2" / "shape-d__rows16384__r1.json").unlink()
        findings = assemble(dest)
        if not any("0 shape-d legs beside 1" in f for f in findings):
            print(f"self-test FAILED: a session with no fleet legs: {findings}", file=sys.stderr)
            return 1
        laid = sorted(p.name for p in (dest / "ladder-K2").glob("*.json"))
        direct = sorted(p.name for p in (dest / "ladder-K2" / "direct").glob("*.json"))
        want_laid = ["placed__rows16384__r1.json", "plan-partitioned__rows16384__r1.json", "shape-d__rows16384__r1.json"]
        want_direct = ["plan-partitioned__rows16384__r1.json", "shape-d__rows16384__r1.json"]
        if laid != want_laid or direct != want_direct:
            print(f"self-test FAILED: the layout: {laid} / {direct}", file=sys.stderr)
            return 1
    print(
        "plane_scaling_verdict self-test: OK — the bar, the layout, the fleet-size refusal "
        "and the empty-session refusal"
    )
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--self-test"]:
        return self_test()
    if len(argv) == 2 and argv[0] == "assemble":
        findings = assemble(Path(argv[1]))
        for finding in findings:
            print(f"::error::{finding}", file=sys.stderr)
        return 1 if findings else 0
    if len(argv) == 2 and argv[0] == "decide":
        return decide(Path(argv[1]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
