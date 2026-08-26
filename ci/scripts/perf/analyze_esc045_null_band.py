#!/usr/bin/env python3
"""esc-045 control (a) -- the NULL BAND of the paired gradient-cosine
statistic (`docs/maintainer/fine-tune-performance-guide.md` section 6,
`.jammi/escapes.jsonl`'s esc-045 row, control (a)): "the identical paired
statistic computed torch-vs-torch (replicate torch runs / seeds at the same
operating points) and jammi-eager-vs-jammi-eager; if a SAME-ARM pairing also
produces a consistent 6/6 sign, the statistic is chaotic and the RED is
vacuous."

Reads `run_esc045_null_band.sh`'s raw dumps and, for EACH of the 6 b4-s128
operating points esc-045's own table covers (seeds 42/43/44 x
{gaussian, poststep}), computes the SAME statistic that table's
`torch_bf16`/`eager` columns report --
`full_tensor_cosine(bf16_arm, jammi_f32_truth)`, the 224-tensor concatenated
cosine `analyze_esc045_torch_column.py`'s own `compare_arm_vs_truth` computes
-- for BOTH bf16 replicates of the torch arm and BOTH bf16 replicates of the
jammi-eager arm.

REPLICATE DESIGN, restated from `run_esc045_null_band.sh`'s own comment:
replicate A and replicate B of one arm at one operating point are two
INDEPENDENT process invocations of the IDENTICAL command -- same seed, same
`--lora-weights-in` shared safetensors file, same batch/config, same
`jammi_f32_truth` dump compared against. Every `RUN_IDENTITY_FIELDS` entry
(`compare_grad_oracle.py`'s determinant table) is pinned equal between A and
B. The ONLY thing that can differ is RUN-TO-RUN NONDETERMINISM in that arm's
own bf16 kernel execution (GPU reduction-order / algorithm-selection
variability) -- which is the SAME CATEGORY of nuisance (floating-point
accumulation-order variability in a bf16 forward+backward) that separates
TWO INDEPENDENTLY-IMPLEMENTED bf16 backward paths (torch's vs jammi's) at a
SHARED operating point. A different LoRA-init draw or a different data seed
was explicitly REJECTED as the replicate nuisance: `grad_oracle.rs`'s module
doc states both real arms in esc-045's own comparison load IDENTICAL LoRA
weights via safetensors and the SAME batch (`synthetic_ids(.., seed + i,
..)`), so varying either between replicate A/B would inject a nuisance the
real between-arm comparison never carries -- an apples-to-oranges null band.

Never reimplements the cosine arithmetic: imports `compare_arm_vs_truth`
straight from `analyze_esc045_torch_column.py` (same directory, same
producer, family F's numpy-first-oracle convention -- a SEPARATE
reimplementation here could silently drift from the statistic the headline
table itself reports, defeating the whole point of a same-metric null band).

RED/GREEN reading (printed by `main()`, never silently inferred): if a
same-arm replicate pairing ALSO produces a consistent 6/6 (or 0/6, i.e. -6/6)
sign over these 6 points, the paired-sign statistic has no resolving power
and esc-045's block-vs-torch 6/6 is vacuous. If same-arm signs are close to
an even split, the block-vs-torch 6/6 stands as real signal (subject to every
other control in esc-045's row). The null band itself (per arm, per point:
|repA - repB|; aggregated as the range over the 6 points) is reported
alongside so it can be compared directly against the block-arm deficits in
section 6's table (e.g. 0.610 vs 0.796, a deficit of 0.186).

Usage:
    python3 analyze_esc045_null_band.py OUT_DIR [--out OUT_DIR/null_band.json]

Self-test (no GPU, no real dump, a synthetic fixture):
    python3 analyze_esc045_null_band.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_esc045_torch_column import compare_arm_vs_truth  # noqa: E402

CONFIGS = [
    ("b4-s128-seed42", 4, 128, 42),
    ("b4-s128-seed43", 4, 128, 43),
    ("b4-s128-seed44", 4, 128, 44),
]
OP_POINTS = ["gaussian", "poststep"]
ARMS = ["torch_bf16", "jammi_bf16_eager"]


def load_dump(raw_dir: Path, name: str) -> dict:
    path = raw_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing raw dump: {path}")
    with open(path) as fh:
        return json.load(fh)


def full_cosine_vs_truth(raw_dir: Path, label: str, op: str, arm: str, rep: str) -> float:
    truth = load_dump(raw_dir, f"{label}__{op}__jammi_f32_truth")
    arm_report = load_dump(raw_dir, f"{label}__{op}__{arm}_{rep}")
    cmp = compare_arm_vs_truth(arm_report, truth)
    return cmp["full_tensor_cosine"]


def sign(x: float) -> str:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def analyze(out_dir: Path, configs=CONFIGS) -> dict:
    raw_dir = out_dir / "raw"
    rows = []
    for label, batch, seq, seed in configs:
        for op in OP_POINTS:
            row = {"config": label, "batch": batch, "seq": seq, "seed": seed, "op_point": op}
            for arm in ARMS:
                cos_a = full_cosine_vs_truth(raw_dir, label, op, arm, "repA")
                cos_b = full_cosine_vs_truth(raw_dir, label, op, arm, "repB")
                diff = cos_a - cos_b
                row[f"{arm}_repA"] = cos_a
                row[f"{arm}_repB"] = cos_b
                row[f"{arm}_paired_diff"] = diff
                row[f"{arm}_sign"] = sign(diff)
                row[f"{arm}_null_band"] = abs(diff)
            rows.append(row)

    summary = {}
    for arm in ARMS:
        signs = [row[f"{arm}_sign"] for row in rows]
        pos = signs.count("+")
        neg = signs.count("-")
        zero = signs.count("0")
        bands = [row[f"{arm}_null_band"] for row in rows]
        summary[arm] = {
            "signs": signs,
            "positive": pos,
            "negative": neg,
            "zero": zero,
            "n": len(signs),
            "null_band_per_point": bands,
            "null_band_min": min(bands),
            "null_band_max": max(bands),
            "null_band_range": max(bands) - min(bands),
        }

    return {"rows": rows, "summary": summary}


def format_table(results: dict) -> str:
    lines = []
    header = (
        f"{'config':<16} {'op':<9} "
        f"{'torch repA':>11} {'torch repB':>11} {'diff':>9} {'sign':>4} "
        f"{'eager repA':>11} {'eager repB':>11} {'diff':>9} {'sign':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in results["rows"]:
        lines.append(
            f"{row['config']:<16} {row['op_point']:<9} "
            f"{row['torch_bf16_repA']:>11.6f} {row['torch_bf16_repB']:>11.6f} "
            f"{row['torch_bf16_paired_diff']:>+9.6f} {row['torch_bf16_sign']:>4} "
            f"{row['jammi_bf16_eager_repA']:>11.6f} {row['jammi_bf16_eager_repB']:>11.6f} "
            f"{row['jammi_bf16_eager_paired_diff']:>+9.6f} {row['jammi_bf16_eager_sign']:>4}"
        )
    lines.append("")
    for arm, s in results["summary"].items():
        lines.append(
            f"{arm}: signs={s['signs']} -> {s['positive']} positive / {s['negative']} negative / "
            f"{s['zero']} zero (n={s['n']}); null band per point (|repA-repB|) "
            f"min={s['null_band_min']:.6f} max={s['null_band_max']:.6f}"
        )
    return "\n".join(lines)


def _self_test() -> int:
    # Synthetic 1-tensor fixture: two "replicate" dumps of the same arm
    # whose gradients differ by a tiny perturbation (simulating GPU
    # reduction-order noise), compared against a shared truth.
    def report(grad):
        return {"gradients": {"t": {"shape": [len(grad)], "grad": grad, "weight": grad}}}

    truth = report([1.0, 0.0, 0.0])
    rep_a = report([1.0, 0.00005, 0.0])
    rep_b = report([1.0, -0.0001, 0.0])
    cos_a = compare_arm_vs_truth(rep_a, truth)["full_tensor_cosine"]
    cos_b = compare_arm_vs_truth(rep_b, truth)["full_tensor_cosine"]
    diff = cos_a - cos_b
    assert abs(diff) > 0.0, "replicate perturbation must produce a nonzero paired diff"
    assert sign(diff) in ("+", "-")
    print("analyze_esc045_null_band: self-test OK")
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
