#!/usr/bin/env python3
"""Merge stage for `ci/scripts/perf/encode_ab.sh`: jammi's `encode-step` legs
against the PyTorch reference's (`crates/jammi-bench/reference/torch_encode.py`),
measured back to back on ONE box over ONE corpus.

Importable (never an inline heredoc, which would have no automated coverage) —
`test_encode_ab.py` drives the REAL entry point (`main`) against leg
directories shaped like the producer's own `.exit`/`.json` output.

## The legs, and the order they run in

Four arms, each run twice, as a PALINDROME:

    jammi-p1  jammi-pN  torch-corpus  torch-sorted | torch-sorted-2  torch-corpus-2  jammi-pN-2  jammi-p1-2

`jammi-p1`/`jammi-pN` are `encode-step` at `[inference] partitions` 1 and N;
`torch-corpus` is the reference forwarding the rows in the engine's own order
(the semantic twin); `torch-sorted` is the reference forwarding them
longest-first, as `sentence-transformers`' `encode()` does (the bar a user
holds the engine to). In a palindrome every arm's two runs are centred on the
same instant, so a box that drifts over the run (a thermal or clock trend that
scales every wall time by a growing factor) moves every arm's mean alike, and
the first-order term cancels out of the ratio of ANY two arms — the A, B, B, A
order `gpu_inference_ab.py` documents, for four arms at once.

## What is refused, and what is only recorded

* `INVALID` — the legs did not measure the same thing. Every pair below is
  checked through `ab_merge.generic_leg_identity_fields`/
  `generic_leg_premise_violations`, the one refusal core every producer
  shares: an arm's two runs, and the two jammi arms, on
  `identity_fields.ENCODE_IDENTITY_FIELDS`; every torch leg against a jammi leg
  on `ENCODE_TWIN_IDENTITY_FIELDS` (the subset a producer that read the corpus
  from a file can state). A leg whose own record contradicts its label
  (`jammi-pN` not at the run's N, `torch-sorted` not length-sorted, a torch
  leg that did or did not build the ANN graph the run says it should) is
  refused the same way: `partitions`, `order` and `ann_index` are what the arms
  differ by, provenance on their reports, so identity cannot speak for them.
* `INVALID_MEASUREMENT` — they measured the same thing and produced different
  things: the jammi arms' persisted vectors are not byte-identical (the
  engine contracts `partitions` never to change the written bytes), or a torch
  leg's per-row cosine against the jammi vectors falls under the floor. A speed
  ratio between two stacks that embedded different vectors is not recorded as
  one.
* `INCOMPLETE` / `DRY_RUN` — a leg failed or is missing / the run was a dry
  run. Nothing is compared.
* `GREEN` — the premises and outcomes hold, and the comparisons are RECORDED.
  A parity verdict never changes the status or the exit code: "is jammi as
  fast and as small as PyTorch here" is the question a run answers, and
  "no" is an answer, not a broken run.

## The estimator least favourable to jammi

Each comparison sets a jammi arm beside a torch arm, per row count. With two
runs of each there are four ways to pair them; the recorded `conservative`
ratio is the least favourable of the four to jammi and `optimistic` the most —
for a rate (rows/s, tokens/s; higher is better) `min(jammi) / max(torch)`, for
a cost (peak RSS, peak VRAM growth, the fitted fixed and per-row
milliseconds; lower is better) `max(jammi) / min(torch)` — the same "ratio
uses the min of two torch runs" discipline `ab_merge.bar_ratio_classification`
applies. The rows/s verdict is `PASS` when even the conservative ratio clears
the bar, `FAIL` when even the optimistic one does not, and `INDETERMINATE`
when the runs disagree about which side of the bar the arm is on.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_merge  # noqa: E402
from gpu_inference_ab import load_leg  # noqa: E402
from identity_fields import (  # noqa: E402
    ENCODE_IDENTITY_FIELDS,
    ENCODE_NULL_IS_A_VALUE_FIELDS,
    ENCODE_TWIN_IDENTITY_FIELDS,
)

JAMMI_ARMS = ("jammi-p1", "jammi-pN")
TORCH_ARMS = ("torch-corpus", "torch-sorted")
ARMS = JAMMI_ARMS + TORCH_ARMS
SECOND_RUN_SUFFIX = "-2"
# The palindrome — `encode_ab.sh` runs exactly this, in exactly this order
# (`test_encode_ab_sh_dry_run.py` holds the two together).
LEG_ORDER = ARMS + tuple(arm + SECOND_RUN_SUFFIX for arm in reversed(ARMS))

# What each torch arm's own report must say it was.
TORCH_ARM_ORDER = {"torch-corpus": "corpus", "torch-sorted": "length-sorted"}

# The parity bar a rows/s ratio is classified against, and the cosine a torch
# leg's every row must reach against the jammi vectors. `encode_ab.sh` writes
# the run's values — with `partitions` (the jammi-pN arm's N) and
# `torch_ann_index` (whether the torch legs close their span on the ANN graph
# the engine's sink builds) — into `raw_dir` before any leg runs (the file-based
# state-passing convention `gpu_inference_ab.py`'s `mode` marker uses).
DEFAULT_PASS_RATIO = 0.9
DEFAULT_COSINE_FLOOR = 0.99

RATES = ("rows_per_s", "tokens_per_s")
COSTS = ("peak_rss_bytes", "peak_vram_delta_bytes")
FIT_COSTS = ("fixed_ms", "per_row_ms")


def runs_of(arm):
    return (arm, arm + SECOND_RUN_SUFFIX)


def read_marker(raw_dir, name, default, parse):
    path = os.path.join(raw_dir, name)
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return parse(fh.read().strip())


def tier_of(leg, report):
    """The `encode_step` block of either producer's report."""
    if leg.startswith("jammi"):
        return (report.get("tiers") or {}).get("encode_step") or {}
    return report.get("encode_step") or {}


def label_violations(leg, tier, partitions_n, torch_ann_index):
    """A leg's own record against what its label, and this run, claim it
    was."""
    arm = leg.removesuffix(SECOND_RUN_SUFFIX)
    if arm in JAMMI_ARMS:
        claimed = {"partitions": 1 if arm == "jammi-p1" else partitions_n}
    else:
        claimed = {"order": TORCH_ARM_ORDER[arm], "ann_index": torch_ann_index}
    return [
        f"{leg} recorded {field}={tier.get(field)!r}, this run's {leg} means {expected!r}"
        for field, expected in claimed.items()
        if tier.get(field) != expected
    ]


def premise_violations(tiers, leg_a, leg_b, fields):
    identity = lambda leg: ab_merge.generic_leg_identity_fields(  # noqa: E731
        tiers[leg], fields, ENCODE_NULL_IS_A_VALUE_FIELDS
    )
    return ab_merge.generic_leg_premise_violations(fields, identity(leg_a), identity(leg_b), leg_a, leg_b)


def points_by_rows(tier):
    return {point["rows"]: point for point in tier.get("points") or []}


def outcome_violations(tiers, cosine_floor):
    """Same premise, different product — see the module doc's
    `INVALID_MEASUREMENT`."""
    violations = []
    digests = {leg: {r: p.get("vectors_digest") for r, p in points_by_rows(tiers[leg]).items()} for leg in tiers if leg.startswith("jammi")}
    reference_leg = JAMMI_ARMS[0]
    for leg, by_rows in digests.items():
        if by_rows != digests[reference_leg]:
            violations.append(
                f"{leg} persisted different vectors than {reference_leg} over the same corpus: "
                f"{by_rows} vs {digests[reference_leg]}"
            )
    for leg in (leg for leg in tiers if leg.startswith("torch")):
        for rows, point in points_by_rows(tiers[leg]).items():
            agreement = point.get("agreement")
            if agreement is None:
                violations.append(f"{leg} @ {rows} rows compared its vectors against no jammi vectors")
            elif not agreement["cosine_min"] >= cosine_floor:
                violations.append(
                    f"{leg} @ {rows} rows: cosine_min {agreement['cosine_min']!r} against the jammi vectors "
                    f"is under the floor {cosine_floor}"
                )
    return violations


def ratio_record(jammi_values, torch_values, higher_is_better):
    """The four-way pairing of two jammi runs with two torch runs — see the
    module doc's "The estimator least favourable to jammi". `None` when
    either side measured nothing (a CPU leg's VRAM)."""
    if any(v is None for v in jammi_values + torch_values) or not all(v > 0 for v in torch_values + jammi_values):
        return None
    worst, best = (min, max) if higher_is_better else (max, min)
    return {
        "jammi": jammi_values,
        "torch": torch_values,
        "conservative": worst(jammi_values) / best(torch_values),
        "optimistic": best(jammi_values) / worst(torch_values),
    }


def parity_verdict(record, pass_ratio):
    if record is None:
        return None
    if record["conservative"] >= pass_ratio:
        return "PASS"
    return "FAIL" if record["optimistic"] < pass_ratio else "INDETERMINATE"


def compare(tiers, jammi_arm, torch_arm, pass_ratio):
    """One jammi arm beside one torch arm: every measured quantity, per row
    count, and the fitted cost split."""
    jammi_runs, torch_runs = runs_of(jammi_arm), runs_of(torch_arm)
    per_rows = {}
    for rows in points_by_rows(tiers[jammi_arm]):
        values = lambda legs, key: [points_by_rows(tiers[leg])[rows].get(key) for leg in legs]  # noqa: E731
        entry = {key: ratio_record(values(jammi_runs, key), values(torch_runs, key), True) for key in RATES}
        entry.update({key: ratio_record(values(jammi_runs, key), values(torch_runs, key), False) for key in COSTS})
        entry["parity"] = parity_verdict(entry["rows_per_s"], pass_ratio)
        per_rows[str(rows)] = entry
    fit = lambda legs, key: [(tiers[leg].get("fit_p50") or {}).get(key) for leg in legs]  # noqa: E731
    return {
        "per_rows": per_rows,
        "fit_p50": {key: ratio_record(fit(jammi_runs, key), fit(torch_runs, key), False) for key in FIT_COSTS},
    }


def leg_record(leg, entry):
    record = {"outcome": entry["outcome"]}
    if entry["outcome"] != "OK":
        return record
    tier = tier_of(leg, entry["report"])
    identity_fields = ENCODE_IDENTITY_FIELDS if leg.startswith("jammi") else ENCODE_TWIN_IDENTITY_FIELDS
    record["identity"] = {k: tier.get(k) for k in identity_fields}
    record["provenance"] = {k: v for k, v in tier.items() if k not in identity_fields and k not in ("points", "fit_p50", "fit_min")}
    record["provenance"]["report"] = entry["report"].get("provenance")
    record.update({k: tier.get(k) for k in ("points", "fit_p50", "fit_min")})
    return record


def build_report(raw_dir, git_sha=None):
    """`(merged, exit_code)` — see the module doc for the status lattice."""
    partitions_n = read_marker(raw_dir, "partitions", None, int)
    torch_ann_index = read_marker(raw_dir, "torch_ann_index", None, lambda raw: raw == "1")
    pass_ratio = read_marker(raw_dir, "pass_ratio", DEFAULT_PASS_RATIO, float)
    cosine_floor = read_marker(raw_dir, "cosine_floor", DEFAULT_COSINE_FLOOR, float)
    entries = {leg: load_leg(raw_dir, leg) for leg in LEG_ORDER}
    merged = {
        "schema_version": 2,
        "git_sha": git_sha,
        "box": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "producer": {
            "path": "ci/scripts/perf/encode_ab.sh",
            "kind": "script",
            "invocation": "ci/scripts/perf/encode_ab.sh",
            "gating": "none",
        },
        "leg_order": list(LEG_ORDER),
        "partitions_n": partitions_n,
        "torch_ann_index": torch_ann_index,
        "pass_ratio": pass_ratio,
        "cosine_floor": cosine_floor,
        "identity_fields": list(ENCODE_IDENTITY_FIELDS),
        "twin_identity_fields": list(ENCODE_TWIN_IDENTITY_FIELDS),
        "legs": {leg: leg_record(leg, entry) for leg, entry in entries.items()},
        "leg_premise_violations": [],
        "outcome_violations": [],
        "comparisons": {},
    }
    outcomes = {entry["outcome"] for entry in entries.values()}
    if outcomes != {"OK"}:
        merged["status"] = "DRY_RUN" if outcomes == {"DRY_RUN"} else "INCOMPLETE"
        return merged, 0 if merged["status"] == "DRY_RUN" else 1

    tiers = {leg: tier_of(leg, entry["report"]) for leg, entry in entries.items()}
    violations = [
        v for leg in LEG_ORDER for v in label_violations(leg, tiers[leg], partitions_n, torch_ann_index)
    ]
    for arm in ARMS:
        fields = ENCODE_IDENTITY_FIELDS if arm in JAMMI_ARMS else ENCODE_TWIN_IDENTITY_FIELDS
        violations += premise_violations(tiers, *runs_of(arm), fields)
    violations += premise_violations(tiers, *JAMMI_ARMS, ENCODE_IDENTITY_FIELDS)
    for torch_arm in TORCH_ARMS:
        violations += premise_violations(tiers, JAMMI_ARMS[0], torch_arm, ENCODE_TWIN_IDENTITY_FIELDS)
    merged["leg_premise_violations"] = violations
    if violations:
        merged["status"] = "INVALID"
        return merged, 1

    merged["outcome_violations"] = outcome_violations(tiers, cosine_floor)
    if merged["outcome_violations"]:
        merged["status"] = "INVALID_MEASUREMENT"
        return merged, 1

    merged["comparisons"] = {
        f"{jammi_arm} vs {torch_arm}": compare(tiers, jammi_arm, torch_arm, pass_ratio)
        for jammi_arm in JAMMI_ARMS
        for torch_arm in TORCH_ARMS
    }
    merged["status"] = "GREEN"
    return merged, 0


def render_table(merged):
    lines = [f"status={merged['status']}"]
    lines += [f"  premise: {v}" for v in merged["leg_premise_violations"]]
    lines += [f"  outcome: {v}" for v in merged["outcome_violations"]]
    fmt = lambda record: "n/a" if record is None else f"{record['conservative']:.3f}..{record['optimistic']:.3f}"  # noqa: E731
    for name, comparison in merged["comparisons"].items():
        lines.append(f"{name}  (jammi/torch, conservative..optimistic)")
        for rows, entry in comparison["per_rows"].items():
            lines.append(
                f"  rows={rows:>7}  rows/s {fmt(entry['rows_per_s'])} [{entry['parity']}]  "
                f"rss {fmt(entry['peak_rss_bytes'])}  vram {fmt(entry['peak_vram_delta_bytes'])}"
            )
        fit = comparison["fit_p50"]
        lines.append(f"  fit            fixed_ms {fmt(fit['fixed_ms'])}  per_row_ms {fmt(fit['per_row_ms'])}")
    for arm in JAMMI_ARMS:
        fit = (merged["legs"].get(arm) or {}).get("fit_p50")
        if fit:
            lines.append(f"{arm}: fixed_ms={fit['fixed_ms']:.3f} per_row_ms={fit['per_row_ms']:.5f}")
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print("usage: encode_ab.py RAW_DIR OUT_DIR [GIT_SHA]", file=sys.stderr)
        return 2
    raw_dir, out_dir = argv[0], argv[1]
    merged, exit_code = build_report(raw_dir, git_sha=argv[2] if len(argv) > 2 else None)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "encode_ab_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    print(f"=== merged report: {out_path} ===")
    print(render_table(merged))
    print(f"=== exit={exit_code} ===")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
