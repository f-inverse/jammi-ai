#!/usr/bin/env python3
"""Put a committed baseline out of reach, in place.

The teeth-proof half of the workflow-layer gates (perf.yml): with the committed
number made unreachable the tier MUST exit non-zero — proving the exit-code
gate is wired through, not decorative. The caller perturbs an ephemeral CI
checkout only and restores the file afterwards.

A committed RATE (`baseline_pairs_per_s`) is inflated 100x, so the derived
floor (`baseline * (1 - 0.30)`) is unreachable. A committed COST budget (the
`per_row_ratio` / `fixed_rows` of a `*_overhead` block) is deflated 100x, so
the derived ceiling (`budget / (1 - 0.30)`) is.

Usage: inflate_baseline.py <path-to-baseline.json>
"""

import json
import sys

FACTOR = 100

p = sys.argv[1]
d = json.load(open(p))
moved = []
if "baseline_pairs_per_s" in d:
    d["baseline_pairs_per_s"] *= FACTOR
    moved.append(f"baseline_pairs_per_s -> {d['baseline_pairs_per_s']} (floor now unreachable)")
for name, block in d.items():
    if name.endswith("_overhead"):
        for cost in block:
            block[cost] /= FACTOR
        moved.append(f"{name} -> {block} (ceiling now unreachable)")
if not moved:
    sys.exit(f"{p} commits no rate baseline and no overhead budget this script knows how to move")
json.dump(d, open(p, "w"), indent=2)
print("\n".join(moved))
