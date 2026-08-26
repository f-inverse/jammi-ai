#!/usr/bin/env python3
"""The numpy-first comparator for `run_esc045_torch_column.sh`'s raw dumps
(esc-045, ledger rows 258/260 + the #383 audit addendum) — reads every
`OUT_DIR/raw/<label>__<op_point>__<arm>.json` the runner script produced and
computes, per `(label, op_point, arm)`:

  - the FULL-224-tensor cosine (every matched tensor's gradient elements
    concatenated into one vector, one cosine) — the metric ledger rows
    258/260 found NOT seed/shape-stable (b4·s128 seed42: all-fused 0.610 /
    seed43: 0.258).
  - the MEDIAN per-tensor cosine (224 individual cosines, one per matched
    tensor, median taken) — because layers 0-4 carry 84% of the gradient's
    L2² mass (a prior finding this script does not re-derive), the
    full-tensor number above is dominated by a handful of large early-layer
    tensors; the median gives the "typical tensor" a vote regardless of its
    size.
  - the MASS-WEIGHTED MEAN per-tensor cosine, weighted by each tensor's own
    L2² norm on the TRUTH side of the comparison (`sum(truth_grad ** 2)`) —
    a middle ground between the size-dominated full-tensor number and the
    size-blind median.

against BOTH `jammi_f32_truth` and `torch_f32_truth` (recorded separately,
never averaged — the whole point is telling apart "jammi's bf16 arithmetic
diverges from a REAL reference" from "the metric itself is chaotic at this
seed/shape", which requires seeing both reference cosines side by side).

Never transcribes a number: every cosine below is computed HERE, from the
raw per-element gradient arrays the runner script's `jammi-bench grad-oracle`/
`torch_grad_oracle.py` invocations wrote — this file is the numpy-first
oracle Family F's own principle names (`docs/maintainer/cuda-kernel-guide.md`
never asserted against a value from `run_grad_oracle_ablation.sh`'s
Rust-computed `overall_cosine_vs_f32_truth`, which is deliberately a
DIFFERENT producer's own arithmetic — see `grad_oracle_ablation.rs`'s
module doc's "which side computes the cosine" section for why that module
computes in Rust and this one recomputes independently in Python instead of
trusting it).

Usage:
    python3 analyze_esc045_torch_column.py OUT_DIR --out OUT_DIR/report.json

Self-test (no GPU, no real dump, a synthetic 3-tensor fixture instead):
    python3 analyze_esc045_torch_column.py --self-test
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a committed transitive dep
    HAVE_NUMPY = False

VACUOUS_NORM_FLOOR = 1e-12

CONFIGS = [
    ("b4-s128-seed42", 4, 128, 42),
    ("b4-s128-seed43", 4, 128, 43),
    ("b4-s128-seed44", 4, 128, 44),
    ("b8-s128-seed42", 8, 128, 42),
    ("b8-s512-seed42", 8, 512, 42),
]
OP_POINTS = ["gaussian", "poststep"]
JAMMI_ARMS = ["jammi_bf16_fused", "jammi_bf16_eager"]
TORCH_ARMS = ["torch_bf16"]
TRUTHS = ["jammi_f32_truth", "torch_f32_truth"]


def _cosine(a, b) -> float:
    """`float64` dot product / norm — the SAME `VACUOUS_NORM_FLOOR`-gated
    non-vacuous control every sibling comparator in this crate (`compare_
    grad_oracle.py`'s `cosine_similarity`, `grad_oracle_ablation.rs`'s
    `cosine`) applies: a near-zero denominator returns `0.0`, never `NaN`
    or a `ZeroDivisionError` (family F — a zero vector on either side must
    fail every possible way, including non-finite, not just the ordinary
    threshold check).
    """
    if HAVE_NUMPY:
        af = np.asarray(a, dtype=np.float64)
        bf = np.asarray(b, dtype=np.float64)
        na = float(np.sqrt(np.dot(af, af)))
        nb = float(np.sqrt(np.dot(bf, bf)))
        denom = na * nb
        if denom < VACUOUS_NORM_FLOOR:
            return 0.0
        return float(np.dot(af, bf) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    denom = na * nb
    if denom < VACUOUS_NORM_FLOOR:
        return 0.0
    return dot / denom


def _l2sq(a) -> float:
    if HAVE_NUMPY:
        af = np.asarray(a, dtype=np.float64)
        return float(np.dot(af, af))
    return sum(x * x for x in a)


def _has_nonfinite(a) -> bool:
    if HAVE_NUMPY:
        return bool(np.isnan(np.asarray(a, dtype=np.float64)).any() or np.isinf(np.asarray(a, dtype=np.float64)).any())
    return any((x != x) or x in (float("inf"), float("-inf")) for x in a)


def load_dump(raw_dir: Path, label: str, op_point: str, arm: str) -> dict:
    path = raw_dir / f"{label}__{op_point}__{arm}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing raw dump: {path}")
    with open(path) as fh:
        return json.load(fh)


def compare_arm_vs_truth(arm_report: dict, truth_report: dict) -> dict:
    """Returns `{full_tensor_cosine, median_cosine, mass_weighted_mean_cosine,
    matched_tensor_count, per_tensor}` — refuses (raises) on a non-finite
    gradient element on either side, mirroring `grad_oracle_ablation.rs`'s
    `build_arm`'s own refusal (family F: "write comparisons affirmatively",
    never let NaN-poisoned data silently produce a low-but-plausible
    number).
    """
    arm_grads = arm_report["gradients"]
    truth_grads = truth_report["gradients"]
    names = sorted(set(arm_grads) & set(truth_grads))
    missing = sorted(set(arm_grads) ^ set(truth_grads))
    if missing:
        raise ValueError(f"tensor name sets differ between arm and truth: {missing}")

    all_a: list[float] = []
    all_b: list[float] = []
    per_tensor = []
    for name in names:
        a = arm_grads[name]["grad"]
        b = truth_grads[name]["grad"]
        if _has_nonfinite(a) or _has_nonfinite(b):
            raise ValueError(f"tensor {name!r}: non-finite gradient element on at least one side")
        cos = _cosine(a, b)
        mass = _l2sq(b)  # mass measured on the TRUTH side, by construction
        per_tensor.append({"name": name, "cosine": cos, "truth_mass_l2sq": mass, "n": len(a)})
        all_a.extend(a)
        all_b.extend(b)

    full_cosine = _cosine(all_a, all_b)
    cosines = [row["cosine"] for row in per_tensor]
    median_cosine = statistics.median(cosines)
    total_mass = sum(row["truth_mass_l2sq"] for row in per_tensor)
    if total_mass < VACUOUS_NORM_FLOOR:
        mass_weighted_mean = 0.0
    else:
        mass_weighted_mean = sum(row["cosine"] * row["truth_mass_l2sq"] for row in per_tensor) / total_mass

    return {
        "full_tensor_cosine": full_cosine,
        "median_per_tensor_cosine": median_cosine,
        "mass_weighted_mean_cosine": mass_weighted_mean,
        "matched_tensor_count": len(names),
        "per_tensor": per_tensor,
    }


def requested_equals_fired(report: dict) -> dict:
    requested = sorted(report.get("kernels_disabled_requested", []))
    fired = sorted(report.get("kernels_disabled_fired", []))
    return {
        "requested": requested,
        "fired": fired,
        "requested_count": len(requested),
        "fired_count": len(fired),
        "equal": requested == fired,
    }


def analyze(out_dir: Path, configs=CONFIGS) -> dict:
    raw_dir = out_dir / "raw"
    results = {"configs": []}
    for label, batch, seq, seed in configs:
        config_row = {"label": label, "batch": batch, "seq": seq, "seed": seed, "op_points": {}}
        for op_point in OP_POINTS:
            truths = {t: load_dump(raw_dir, label, op_point, t) for t in TRUTHS}

            # jammi-f32-truth vs torch-f32-truth: the cross-producer sanity
            # check the dispatch asked to be recorded explicitly, computed
            # the SAME way every other cell below is (never a special-cased
            # shortcut).
            f32_cross = compare_arm_vs_truth(truths["jammi_f32_truth"], truths["torch_f32_truth"])
            del f32_cross["per_tensor"]  # keep the committed artifact small; full detail stays in raw/

            arms_row = {}
            for arm in JAMMI_ARMS + TORCH_ARMS:
                arm_report = load_dump(raw_dir, label, op_point, arm)
                vs = {}
                for truth_name, truth_report in truths.items():
                    cmp = compare_arm_vs_truth(arm_report, truth_report)
                    del cmp["per_tensor"]
                    vs[truth_name] = cmp
                arm_entry = {"vs": vs}
                if arm in JAMMI_ARMS:
                    arm_entry["kernels_disabled"] = requested_equals_fired(arm_report)
                arms_row[arm] = arm_entry

            config_row["op_points"][op_point] = {
                "jammi_f32_truth_vs_torch_f32_truth": f32_cross,
                "arms": arms_row,
            }
        results["configs"].append(config_row)
    return results


def format_table(results: dict) -> str:
    lines = []
    header = (
        f"{'config':<16} {'op':<9} {'arm':<17} {'vs':<15} "
        f"{'full':>10} {'median':>10} {'mass-wt':>10} {'n':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for cfg in results["configs"]:
        label = cfg["label"]
        for op_point, op_row in cfg["op_points"].items():
            cross = op_row["jammi_f32_truth_vs_torch_f32_truth"]
            lines.append(
                f"{label:<16} {op_point:<9} {'(f32 x-check)':<17} {'jammi-vs-torch':<15} "
                f"{cross['full_tensor_cosine']:>10.6f} {cross['median_per_tensor_cosine']:>10.6f} "
                f"{cross['mass_weighted_mean_cosine']:>10.6f} {cross['matched_tensor_count']:>5}"
            )
            for arm, arm_row in op_row["arms"].items():
                for truth_name, cmp in arm_row["vs"].items():
                    lines.append(
                        f"{label:<16} {op_point:<9} {arm:<17} {truth_name:<15} "
                        f"{cmp['full_tensor_cosine']:>10.6f} {cmp['median_per_tensor_cosine']:>10.6f} "
                        f"{cmp['mass_weighted_mean_cosine']:>10.6f} {cmp['matched_tensor_count']:>5}"
                    )
                if "kernels_disabled" in arm_row:
                    kd = arm_row["kernels_disabled"]
                    lines.append(
                        f"{'':<16} {'':<9} {arm:<17} requested={kd['requested_count']} "
                        f"fired={kd['fired_count']} equal={kd['equal']}"
                    )
    return "\n".join(lines)


def _self_test() -> int:
    import math

    # Synthetic 2-tensor fixture: tensor "a" carries most of the mass and a
    # perfect cosine; tensor "b" carries little mass and an ORTHOGONAL
    # (cosine 0) gradient -- pins that the mass-weighted mean tracks "a"
    # closely while the median (equal-weight over 2 tensors) sits at the
    # arithmetic mean of 1.0 and 0.0.
    def report(a_grad, b_grad):
        return {
            "gradients": {
                "a": {"shape": [len(a_grad)], "grad": a_grad, "weight": a_grad},
                "b": {"shape": [len(b_grad)], "grad": b_grad, "weight": b_grad},
            }
        }

    truth = report([10.0, 0.0], [0.0, 1.0])
    arm = report([10.0, 0.0], [1.0, 0.0])  # tensor "b" now orthogonal to truth's "b"
    cmp = compare_arm_vs_truth(arm, truth)
    assert abs(cmp["median_per_tensor_cosine"] - 0.5) < 1e-9, cmp
    assert cmp["mass_weighted_mean_cosine"] > 0.99, cmp  # dominated by "a"'s mass (100 vs 1)
    assert math.isclose(cmp["full_tensor_cosine"], (100.0) / (math.sqrt(101) * math.sqrt(101)), rel_tol=1e-9)

    zero_report = report([0.0, 0.0], [0.0, 0.0])
    zero_cmp = compare_arm_vs_truth(zero_report, zero_report)
    assert zero_cmp["full_tensor_cosine"] == 0.0
    assert zero_cmp["mass_weighted_mean_cosine"] == 0.0

    nonfinite = report([float("nan"), 0.0], [0.0, 1.0])
    try:
        compare_arm_vs_truth(nonfinite, truth)
        raise AssertionError("expected ValueError on a non-finite gradient element")
    except ValueError:
        pass

    kd = requested_equals_fired({"kernels_disabled_requested": ["b", "a"], "kernels_disabled_fired": ["a", "b"]})
    assert kd["equal"] and kd["requested"] == ["a", "b"]
    kd_bad = requested_equals_fired({"kernels_disabled_requested": ["a"], "kernels_disabled_fired": []})
    assert not kd_bad["equal"]

    print("analyze_esc045_torch_column: self-test OK" + (" (numpy)" if HAVE_NUMPY else " (pure-python fallback)"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", nargs="?", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="Write the compact JSON report here.")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.out_dir:
        p.error("out_dir is required unless --self-test")

    out_dir = Path(args.out_dir)
    results = analyze(out_dir)
    table = format_table(results)
    print(table)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
